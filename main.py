
import os, time, base64, asyncio, uuid, threading
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_preprocess
import torch
import functools
from ultralytics import YOLO

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ── Patch torch.load ──────────────────────────────────────────────────────────
torch.load = functools.partial(torch.load, weights_only=False)

# ── Paths — UPDATE THESE ──────────────────────────────────────────────────────
RESNET_MODEL_PATH = "resnet50v2_final.h5"  
UPLOAD_DIR        = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
class Config:
    IMG_SIZE              = (224, 224)
    ROI_SIZE              = 224
    VOTE_WINDOW           = 1
    SUSPICIOUS_THRESHOLD  = 0.5
    VOTE_THRESHOLD        = 1
    POSE_CONF_THRESHOLD   = 0.6
    YOLO_POSE_MODEL       = 'yolo11s-pose.pt'
    IOU_THRESHOLD         = 0.3
    MAX_LOST_FRAMES       = 60
    SUPPRESS_IOU_THRESHOLD= 0.5
    KPT_CONF_THRESHOLD    = 0.4
    BODY_BONE_THICKNESS   = 3
    HEAD_BONE_THICKNESS   = 2
    UPPER_BODY_KPTS       = {0,1,2,3,4,5,6,7,8,9,10,11,12}
    KEYPOINT_NAMES        = [
        'nose','left_eye','right_eye','left_ear','right_ear',
        'left_shoulder','right_shoulder','left_elbow','right_elbow',
        'left_wrist','right_wrist','left_hip','right_hip'
    ]
    SKELETON_CONNECTIONS  = [
        (0,1),(0,2),(1,3),(2,4),
        (0,5),(0,6),
        (5,6),
        (5,7),(7,9),
        (6,8),(8,10),
        (5,11),(6,12),(11,12)
    ]

config = Config()

# ── Skeleton Generator ────────────────────────────────────────────────────────
class SkeletonGenerator:
    def __init__(self, roi_size=224):
        self.roi_size      = roi_size
        self.dilate_kernel = np.ones((3,3), np.uint8)

    def is_valid_skeleton(self, conf):
        th      = config.KPT_CONF_THRESHOLD
        visible = sum(1 for i in config.UPPER_BODY_KPTS if i < len(conf) and conf[i] > th)
        return visible >= 5 and (conf[5] > th or conf[6] > th)

    def create_skeleton_image(self, keypoints, bbox):
        if len(keypoints) < 13:
            return None
        xy   = keypoints[:, :2]
        conf = keypoints[:, 2]
        if not self.is_valid_skeleton(conf):
            return None
        valid_pts = [xy[i] for i in config.UPPER_BODY_KPTS
                     if i < len(conf) and conf[i] > config.KPT_CONF_THRESHOLD]
        if len(valid_pts) < 3:
            return None
        pts   = np.array(valid_pts)
        min_x, min_y = pts.min(axis=0)
        max_x, max_y = pts.max(axis=0)
        skel_w = max_x - min_x
        skel_h = max_y - min_y
        if skel_w < 5 or skel_h < 5:
            return None
        canvas = np.zeros((self.roi_size, self.roi_size), dtype=np.uint8)
        pad    = 20
        scale  = min((self.roi_size - 2*pad) / skel_w,
                     (self.roi_size - 2*pad) / skel_h)
        cx     = (min_x + max_x) / 2
        cy     = (min_y + max_y) / 2
        for (i, j) in config.SKELETON_CONNECTIONS:
            if i >= len(conf) or j >= len(conf):
                continue
            if conf[i] < config.KPT_CONF_THRESHOLD or conf[j] < config.KPT_CONF_THRESHOLD:
                continue
            p1 = (int((xy[i][0] - cx) * scale + self.roi_size/2),
                  int((xy[i][1] - cy) * scale + self.roi_size/2))
            p2 = (int((xy[j][0] - cx) * scale + self.roi_size/2),
                  int((xy[j][1] - cy) * scale + self.roi_size/2))
            is_head = i in {0,1,2,3,4} and j in {0,1,2,3,4}
            thick   = config.HEAD_BONE_THICKNESS if is_head else config.BODY_BONE_THICKNESS
            cv2.line(canvas, p1, p2, 255, thick)
        canvas = cv2.dilate(canvas, self.dilate_kernel, iterations=1)
        return canvas

    def preprocess_batch(self, skeleton_imgs):
        batch = []
        for img in skeleton_imgs:
            rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            rgb = cv2.resize(rgb, config.IMG_SIZE)
            batch.append(rgb.astype(np.float32))
        batch = np.array(batch)
        return resnet_preprocess(batch)

# ── IoU Tracker ───────────────────────────────────────────────────────────────
class SimpleIoUTracker:
    def __init__(self, iou_threshold=0.3, max_lost=60):
        self.iou_threshold = iou_threshold
        self.max_lost      = max_lost
        self.tracks        = {}
        self.next_id       = 1

    def _iou(self, a, b):
        ax1,ay1,ax2,ay2 = a
        bx1,by1,bx2,by2 = b
        ix1,iy1 = max(ax1,bx1), max(ay1,by1)
        ix2,iy2 = min(ax2,bx2), min(ay2,by2)
        inter   = max(0,ix2-ix1)*max(0,iy2-iy1)
        if inter == 0:
            return 0.0
        area_a  = (ax2-ax1)*(ay2-ay1)
        area_b  = (bx2-bx1)*(by2-by1)
        return inter / (area_a + area_b - inter)

    def _suppress_duplicates(self, detections):
        if len(detections) <= 1:
            return detections
        keep    = []
        used    = [False]*len(detections)
        sorted_d= sorted(enumerate(detections), key=lambda x: -x[1][1])
        for i, (orig_i, (bbox_i, conf_i, kpts_i)) in enumerate(sorted_d):
            if used[orig_i]:
                continue
            keep.append((bbox_i, conf_i, kpts_i))
            for j, (orig_j, (bbox_j, conf_j, kpts_j)) in enumerate(sorted_d):
                if i != j and not used[orig_j]:
                    if self._iou(bbox_i, bbox_j) > config.SUPPRESS_IOU_THRESHOLD:
                        used[orig_j] = True
        return keep

    def update(self, detections):
        detections     = self._suppress_duplicates(detections)
        matched_tracks = []
        used_det       = set()
        for track_id, track in list(self.tracks.items()):
            best_iou  = self.iou_threshold
            best_det  = -1
            for i, (bbox, conf, kpts) in enumerate(detections):
                if i in used_det:
                    continue
                iou = self._iou(track['bbox'], bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_det = i
            if best_det >= 0:
                bbox, conf, kpts          = detections[best_det]
                track['bbox']             = bbox
                track['keypoints']        = kpts
                track['lost']             = 0
                used_det.add(best_det)
                matched_tracks.append((track_id, bbox, kpts))
            else:
                track['lost'] += 1
                if track['lost'] > self.max_lost:
                    del self.tracks[track_id]
        for i, (bbox, conf, kpts) in enumerate(detections):
            if i not in used_det:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {
                    'bbox': bbox, 'keypoints': kpts,
                    'lost': 0, 'votes': deque(maxlen=config.VOTE_WINDOW),
                }
                matched_tracks.append((tid, bbox, kpts))
        return matched_tracks

    def add_vote(self, track_id, vote):
        if track_id in self.tracks:
            self.tracks[track_id].setdefault('votes', deque(maxlen=config.VOTE_WINDOW))
            self.tracks[track_id]['votes'].append(vote)

    def get_decision(self, track_id):
        if track_id not in self.tracks:
            return 'Normal', 0.0
        votes = self.tracks[track_id].get('votes', deque())
        if not votes:
            return 'Pending', 0.0
        ratio = sum(votes) / len(votes)
        label = 'Suspicious' if sum(votes) >= config.VOTE_THRESHOLD else 'Normal'
        return label, ratio

    def get_color(self, track_id, label):
        return {'Suspicious': (0,0,255), 'Normal': (0,255,0)}.get(label, (0,215,255))

# ── Pipeline ──────────────────────────────────────────────────────────────────
class OptimizedPipeline:
    def __init__(self, yolo_model, resnet_model, skeleton_gen):
        self.yolo         = yolo_model
        self.resnet       = resnet_model
        self.skeleton_gen = skeleton_gen
        self.tracker      = SimpleIoUTracker(
            iou_threshold=config.IOU_THRESHOLD,
            max_lost=config.MAX_LOST_FRAMES
        )
        self.frame_count      = 0
        self.total_suspicious = 0
        self.skipped_invalid  = 0
        self.timing = {'yolo':[],'skeleton':[],'resnet':[],'tracking':[],'total':[]}

    def process_frame(self, frame):
        t_total   = time.perf_counter()
        self.frame_count += 1
        t0        = time.perf_counter()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results   = self.yolo(rgb_frame, conf=config.POSE_CONF_THRESHOLD, verbose=False)
        self.timing['yolo'].append(time.perf_counter() - t0)

        detections = []
        if results[0].keypoints is not None and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            kpts  = results[0].keypoints.data.cpu().numpy()
            for b, c, k in zip(boxes, confs, kpts):
                detections.append((b, c, k))

        t0             = time.perf_counter()
        matched_tracks = self.tracker.update(detections)
        self.timing['tracking'].append(time.perf_counter() - t0)

        if not matched_tracks:
            self.timing['skeleton'].append(0)
            self.timing['resnet'].append(0)
            self.timing['total'].append(time.perf_counter() - t_total)
            return frame, []

        t0             = time.perf_counter()
        skeleton_batch = []
        track_info     = []
        skipped_tracks = []

        for track_id, bbox, keypoints in matched_tracks:
            skeleton_img = self.skeleton_gen.create_skeleton_image(keypoints, bbox)
            if skeleton_img is None:
                self.skipped_invalid += 1
                skipped_tracks.append((track_id, bbox))
                continue
            skeleton_batch.append(skeleton_img)
            track_info.append((track_id, bbox))

        self.timing['skeleton'].append(time.perf_counter() - t0)
        detections_info = []

        if skeleton_batch:
            t0                 = time.perf_counter()
            batch_preprocessed = self.skeleton_gen.preprocess_batch(skeleton_batch)
            batch_tensor       = tf.constant(batch_preprocessed, dtype=tf.float32)
            predictions        = self.resnet(batch_tensor, training=False).numpy().flatten()
            self.timing['resnet'].append(time.perf_counter() - t0)

            for i, (track_id, bbox) in enumerate(track_info):
                prob          = float(predictions[i]) if i < len(predictions) else 0.0
                is_suspicious = 1 if prob >= config.SUSPICIOUS_THRESHOLD else 0
                self.tracker.add_vote(track_id, is_suspicious)
                label, vote_ratio = self.tracker.get_decision(track_id)
                if label == 'Suspicious':
                    self.total_suspicious += 1
                detections_info.append({
                    'student_id' : track_id,
                    'bbox'       : bbox,
                    'label'      : label,
                    'resnet_prob': prob,
                    'vote_ratio' : vote_ratio,
                })
        else:
            self.timing['resnet'].append(0)

        for track_id, bbox in skipped_tracks:
            label, vote_ratio = self.tracker.get_decision(track_id)
            detections_info.append({
                'student_id' : track_id,
                'bbox'       : bbox,
                'label'      : label,
                'resnet_prob': -1,
                'vote_ratio' : vote_ratio,
            })

        self.timing['total'].append(time.perf_counter() - t_total)
        return frame, detections_info

# ── Drawing ───────────────────────────────────────────────────────────────────
def draw_annotations(frame, detections_info):
    for d in detections_info:
        bbox     = d['bbox']
        label    = d['label']
        prob     = d['resnet_prob']
        track_id = d['student_id']
        color    = (0,0,255) if label == 'Suspicious' else (0,255,0)
        x1,y1,x2,y2 = map(int, bbox)
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        tag = f"ID:{track_id:02d} | {label} ({'skip' if prob < 0 else f'p={prob:.2f}'})"
        tag_y = max(y1-10, 20)
        font_scale = 0.55
        thick = 1
        (tw,th),baseline = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick)
        pad = 5
        cv2.rectangle(frame, (x1, tag_y-th-pad*2), (x1+tw+pad*2, tag_y+baseline), (0,0,0), -1)
        cv2.rectangle(frame, (x1, tag_y-th-pad*2), (x1+tw+pad*2, tag_y+baseline), color, 1)
        cv2.putText(frame, tag, (x1+pad, tag_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thick)
    return frame

def draw_status_bar(frame, frame_num, total_frames, elapsed, inf_fps, n_students, n_sus):
    h, w  = frame.shape[:2]
    bar_h = 36
    bar_y = h - bar_h
    cv2.rectangle(frame, (0,bar_y), (w,h), (15,15,15), -1)
    mins, secs = divmod(int(elapsed), 60)
    progress   = (frame_num / total_frames * 100) if total_frames > 0 else 0
    status = (f'Frame:{frame_num}/{total_frames}  Progress:{progress:.1f}%  '
              f'InferenceFPS:{inf_fps:.1f}  Time:{mins:02d}:{secs:02d}  '
              f'Students:{n_students}  Suspicious:{n_sus}')
    cv2.putText(frame, status, (8, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,220,0), 1, cv2.LINE_AA)
    return frame

# ── App & State ───────────────────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Global models (loaded once)
print("Loading models...")
device      = 'cuda' if torch.cuda.is_available() else 'cpu'
yolo_model  = YOLO('yolo11s-pose.pt').to(device)
resnet_model= load_model(RESNET_MODEL_PATH, compile=False)
skeleton_gen= SkeletonGenerator()
# Warmup
dummy = np.random.rand(1,224,224,3).astype(np.float32)
_ = resnet_model(dummy, training=False)
print(f"Models ready. Device: {device}")

# Job state
jobs: dict = {}  # job_id -> { frames: deque, stats: dict, done: bool, log: list }

PROCESS_EVERY_N = 10

def process_video_thread(job_id: str, video_path: str):
    job = jobs[job_id]
    cap = cv2.VideoCapture(video_path)
    total_f    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pipe       = OptimizedPipeline(yolo_model, resnet_model, skeleton_gen)
    frame_num  = 0
    fps_tracker= deque(maxlen=20)
    start_time = time.time()
    cached_annotations = []
    activity_log       = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        if frame_num % PROCESS_EVERY_N == 1 or PROCESS_EVERY_N == 1:
            t0 = time.time()
            _, dets = pipe.process_frame(frame.copy())
            fps_tracker.append(1.0 / max(time.time()-t0, 1e-9))

            cached_annotations = []
            for d in dets:
                track_id = d['student_id']
                label    = d['label']
                prob     = d['resnet_prob']
                bbox     = pipe.tracker.tracks[track_id]['bbox'] if track_id in pipe.tracker.tracks else None
                color    = pipe.tracker.get_color(track_id, label)
                if bbox is not None:
                    cached_annotations.append((bbox, label, prob, color, track_id))

            mins, secs = divmod(int(time.time()-start_time), 60)
            for bbox, label, prob, color, track_id in cached_annotations:
                if label == 'Suspicious' and prob >= 0:
                    activity_log.append({
                        'time'       : f'{mins:02d}:{secs:02d}',
                        'track_id'   : track_id,
                        'resnet_prob': prob,
                    })

        display_frame = frame.copy()
        dets_for_draw = [
            {'bbox': bbox, 'label': label, 'resnet_prob': prob, 'student_id': tid}
            for bbox, label, prob, color, tid in cached_annotations
        ]
        draw_annotations(display_frame, dets_for_draw)

        elapsed  = time.time() - start_time
        inf_fps  = float(np.mean(fps_tracker)) if fps_tracker else 0
        n_stu    = len(cached_annotations)
        n_sus    = sum(1 for _,l,_,_,_ in cached_annotations if l == 'Suspicious')
        draw_status_bar(display_frame, frame_num, total_f, elapsed, inf_fps, n_stu, n_sus)

        _, buf = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        b64    = base64.b64encode(buf).decode()

        job['frames'].append({
            'b64'         : b64,
            'frame_num'   : frame_num,
            'total_frames': total_f,
            'elapsed'     : elapsed,
            'inf_fps'     : inf_fps,
            'n_students'  : n_stu,
            'n_suspicious': n_sus,
            'activity_log': list(activity_log)[-10:],
            'annotations' : [
                {'track_id': tid, 'label': lbl, 'prob': prb}
                for _, lbl, prb, _, tid in cached_annotations
            ],
        })

        # Keep only last 30 frames in buffer
        while len(job['frames']) > 30:
            job['frames'].popleft()

    cap.release()
    job['done'] = True

# ── Routes ────────────────────────────────────────────────────────────────────
@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    job_id    = str(uuid.uuid4())
    save_path = UPLOAD_DIR / f"{job_id}_{file.filename}"
    with open(save_path, "wb") as f:
        f.write(await file.read())
    jobs[job_id] = {
        'frames': deque(),
        'done'  : False,
        'path'  : str(save_path),
    }
    t = threading.Thread(target=process_video_thread, args=(job_id, str(save_path)), daemon=True)
    t.start()
    return {"job_id": job_id}

@app.get("/stream/{job_id}")
async def stream_frames(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    async def event_generator():
        job        = jobs[job_id]
        last_sent  = -1
        empty_iters= 0
        while True:
            if job['frames']:
                frame_data  = job['frames'].popleft()
                payload     = (
                    f"data: {__import__('json').dumps(frame_data)}\n\n"
                )
                yield payload
                empty_iters = 0
            else:
                if job['done']:
                    yield "data: {\"done\": true}\n\n"
                    break
                empty_iters += 1
                if empty_iters > 300:  # 30s timeout
                    break
                await asyncio.sleep(0.05)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = Path("index.html")
    if html_path.exists():
        return html_path.read_text()
    return "<h1>OEPS Backend Running</h1>"

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
