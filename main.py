import os, time, base64, asyncio, uuid, threading, json
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

# ── Paths ─────────────────────────────────────────────────────────────────────
RESNET_MODEL_PATH = "resnet50v2_final.h5"
UPLOAD_DIR        = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# ── GPU Setup ─────────────────────────────────────────────────────────────────
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"TensorFlow GPUs available: {[g.name for g in gpus]}")

# ── Config ────────────────────────────────────────────────────────────────────
class Config:
    IMG_SIZE               = (224, 224)
    ROI_SIZE               = 224
    VOTE_WINDOW            = 1
    SUSPICIOUS_THRESHOLD   = 0.5
    VOTE_THRESHOLD         = 1
    POSE_CONF_THRESHOLD    = 0.6
    IOU_THRESHOLD          = 0.3
    MAX_LOST_FRAMES        = 60
    SUPPRESS_IOU_THRESHOLD = 0.5
    KPT_CONF_THRESHOLD     = 0.4
    BODY_BONE_THICKNESS    = 3
    HEAD_BONE_THICKNESS    = 2
    UPPER_BODY_KPTS        = {0,1,2,3,4,5,6,7,8,9,10,11,12}
    KEYPOINT_NAMES         = [
        'nose','left_eye','right_eye','left_ear','right_ear',
        'left_shoulder','right_shoulder','left_elbow','right_elbow',
        'left_wrist','right_wrist','left_hip','right_hip'
    ]
    SKELETON_CONNECTIONS   = [
        (0,1),(0,2),(1,3),(2,4),
        (0,5),(0,6),
        (5,6),
        (5,7),(7,9),
        (6,8),(8,10),
        (5,11),(6,12),(11,12)
    ]
    DISPLAY_FPS            = 30
    INFERENCE_FPS          = 10
    FRAME_BUFFER_SIZE      = 60
    JPEG_QUALITY           = 80

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
        pts            = np.array(valid_pts)
        min_x, min_y   = pts.min(axis=0)
        max_x, max_y   = pts.max(axis=0)
        skel_w         = max_x - min_x
        skel_h         = max_y - min_y
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
        return resnet_preprocess(np.array(batch))

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
        inter   = max(0, ix2-ix1) * max(0, iy2-iy1)
        if inter == 0:
            return 0.0
        area_a = (ax2-ax1) * (ay2-ay1)
        area_b = (bx2-bx1) * (by2-by1)
        return inter / (area_a + area_b - inter)

    def _suppress_duplicates(self, detections):
        if len(detections) <= 1:
            return detections
        keep     = []
        used     = [False] * len(detections)
        sorted_d = sorted(enumerate(detections), key=lambda x: -x[1][1])
        for i, (orig_i, (bbox_i, conf_i, kpts_i)) in enumerate(sorted_d):
            if used[orig_i]:
                continue
            keep.append((bbox_i, conf_i, kpts_i))
            for j, (orig_j, (bbox_j, _, _kj)) in enumerate(sorted_d):
                if i != j and not used[orig_j]:
                    if self._iou(bbox_i, bbox_j) > config.SUPPRESS_IOU_THRESHOLD:
                        used[orig_j] = True
        return keep

    def update(self, detections):
        detections     = self._suppress_duplicates(detections)
        matched_tracks = []
        used_det       = set()
        for track_id, track in list(self.tracks.items()):
            best_iou = self.iou_threshold
            best_det = -1
            for i, (bbox, conf, kpts) in enumerate(detections):
                if i in used_det:
                    continue
                iou = self._iou(track['bbox'], bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_det = i
            if best_det >= 0:
                bbox, conf, kpts   = detections[best_det]
                track['bbox']      = bbox
                track['keypoints'] = kpts
                track['lost']      = 0
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
        label = 'Suspicious' if sum(votes) >= config.VOTE_THRESHOLD else 'Normal'
        ratio = sum(votes) / len(votes)
        return label, ratio

    def get_color(self, label):
        return {'Suspicious': (0,0,255), 'Normal': (0,255,0)}.get(label, (0,215,255))

# ── AI Pipeline ───────────────────────────────────────────────────────────────
class AIPipeline:
    def __init__(self, yolo_model, resnet_infer_fn, skeleton_gen):
        self.yolo         = yolo_model
        self.resnet_infer = resnet_infer_fn
        self.skeleton_gen = skeleton_gen
        self.tracker      = SimpleIoUTracker(
            iou_threshold=config.IOU_THRESHOLD,
            max_lost=config.MAX_LOST_FRAMES
        )

    def process_frame(self, frame):
        rgb_frame  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results    = self.yolo(rgb_frame, conf=config.POSE_CONF_THRESHOLD, verbose=False)
        detections = []
        if results[0].keypoints is not None and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            kpts  = results[0].keypoints.data.cpu().numpy()
            for b, c, k in zip(boxes, confs, kpts):
                detections.append((b, c, k))

        matched_tracks = self.tracker.update(detections)
        if not matched_tracks:
            return []

        skeleton_batch = []
        track_info     = []
        skipped_tracks = []
        for track_id, bbox, keypoints in matched_tracks:
            skel = self.skeleton_gen.create_skeleton_image(keypoints, bbox)
            if skel is None:
                skipped_tracks.append((track_id, bbox))
            else:
                skeleton_batch.append(skel)
                track_info.append((track_id, bbox))

        detections_info = []

        if skeleton_batch:
            preprocessed = self.skeleton_gen.preprocess_batch(skeleton_batch)
            batch_tensor = tf.constant(preprocessed, dtype=tf.float32)
            predictions  = self.resnet_infer(batch_tensor).numpy().flatten()

            for i, (track_id, bbox) in enumerate(track_info):
                prob          = float(predictions[i]) if i < len(predictions) else 0.0
                is_suspicious = 1 if prob >= config.SUSPICIOUS_THRESHOLD else 0
                self.tracker.add_vote(track_id, is_suspicious)
                label, vote_ratio = self.tracker.get_decision(track_id)
                detections_info.append({
                    'student_id' : track_id,
                    'bbox'       : bbox,
                    'label'      : label,
                    'resnet_prob': prob,
                    'vote_ratio' : vote_ratio,
                })

        for track_id, bbox in skipped_tracks:
            label, vote_ratio = self.tracker.get_decision(track_id)
            detections_info.append({
                'student_id' : track_id,
                'bbox'       : bbox,
                'label'      : label,
                'resnet_prob': -1,
                'vote_ratio' : vote_ratio,
            })

        return detections_info

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
        tag       = f"ID:{track_id:02d} | {label} ({'skip' if prob < 0 else f'p={prob:.2f}'})"
        tag_y     = max(y1-10, 20)
        fs        = 0.55
        (tw,th), baseline = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        pad = 5
        cv2.rectangle(frame, (x1, tag_y-th-pad*2), (x1+tw+pad*2, tag_y+baseline), (0,0,0), -1)
        cv2.rectangle(frame, (x1, tag_y-th-pad*2), (x1+tw+pad*2, tag_y+baseline), color, 1)
        cv2.putText(frame, tag, (x1+pad, tag_y), cv2.FONT_HERSHEY_SIMPLEX, fs, (255,255,255), 1)
    return frame



# ── Global Model Loading ──────────────────────────────────────────────────────
print("Loading models...")
device       = 'cuda' if torch.cuda.is_available() else 'cpu'
yolo_model   = YOLO('yolo11s-pose.pt').to(device)
resnet_model = load_model(RESNET_MODEL_PATH, compile=False)
skeleton_gen = SkeletonGenerator()

@tf.function(
    input_signature=[tf.TensorSpec(shape=[None, 224, 224, 3], dtype=tf.float32)],
    reduce_retracing=True
)
def resnet_infer(x):
    return resnet_model(x, training=False)

print("Warming up ResNet...")
for bs in [1, 2, 4, 8]:
    dummy = tf.random.normal([bs, 224, 224, 3])
    _     = resnet_infer(dummy)
print(f"Models ready. Device: {device}")

# ── Job State ─────────────────────────────────────────────────────────────────
jobs: dict = {}

def make_job(is_webcam=False):
    return {
        'frames'         : deque(),
        'done'           : False,
        'running'        : True,
        'paused'         : False,
        'is_webcam'      : is_webcam,
        'frame_lock'     : threading.Lock(),
        'result_lock'    : threading.Lock(),
        'new_frame'      : threading.Event(),
        'latest_frame'   : None,
        'latest_results' : None,
        'total_frames'   : 0,
        'frame_num'      : 0,
        'start_time'     : time.time(),
        'activity_log'   : [],
        'inf_fps'        : 0.0,
        'display_fps'    : 0.0,
    }

# ── Thread 1: Capture Loop ────────────────────────────────────────────────────
def capture_loop(job_id: str, source):
    job             = jobs[job_id]
    cap             = cv2.VideoCapture(source)
    is_webcam       = job['is_webcam']
    target_interval = 1.0 / config.DISPLAY_FPS
    frame_num       = 0

    if not is_webcam:
        job['total_frames'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while job['running']:
        if job['paused']:
            time.sleep(0.05)
            continue

        t_start    = time.perf_counter()
        ret, frame = cap.read()

        if not ret:
            if is_webcam:
                time.sleep(0.01)
                continue
            else:
                job['done'] = True
                break

        frame_num += 1
        with job['frame_lock']:
            job['latest_frame'] = frame.copy()
            job['frame_num']    = frame_num

        job['new_frame'].set()

        elapsed_read = time.perf_counter() - t_start
        sleep_time   = target_interval - elapsed_read
        if sleep_time > 0:
            time.sleep(sleep_time)

    cap.release()
    job['running'] = False

# ── Thread 2: Inference Loop ──────────────────────────────────────────────────
def inference_loop(job_id: str):
    job             = jobs[job_id]
    pipe            = AIPipeline(yolo_model, resnet_infer, skeleton_gen)
    target_interval = 1.0 / config.INFERENCE_FPS
    inf_fps_tracker = deque(maxlen=20)

    while job['running'] or not job['done']:
        got_new = job['new_frame'].wait(timeout=1.0)
        if not got_new:
            if job['done']:
                break
            continue

        job['new_frame'].clear()

        if job['paused']:
            continue

        frame_copy = None
        with job['frame_lock']:
            if job['latest_frame'] is not None:
                frame_copy = job['latest_frame'].copy()

        if frame_copy is None:
            continue

        t0              = time.perf_counter()
        detections_info = pipe.process_frame(frame_copy)
        elapsed         = time.perf_counter() - t0

        if elapsed > 0:
            inf_fps_tracker.append(1.0 / elapsed)
        job['inf_fps'] = float(np.mean(inf_fps_tracker)) if inf_fps_tracker else 0.0

        start_time = job.get('start_time', time.time())
        mins, secs = divmod(int(time.time() - start_time), 60)
        for d in detections_info:
            if d['label'] == 'Suspicious' and d['resnet_prob'] >= 0:
                job['activity_log'].append({
                    'time'       : f'{mins:02d}:{secs:02d}',
                    'track_id'   : d['student_id'],
                    'resnet_prob': d['resnet_prob'],
                })
                if len(job['activity_log']) > 100:
                    job['activity_log'].pop(0)

        with job['result_lock']:
            job['latest_results'] = detections_info

        sleep_time = target_interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

# ── Thread 3: Display Loop ────────────────────────────────────────────────────
def display_loop(job_id: str):
    
    job              = jobs[job_id]
    is_webcam        = job['is_webcam']
    target_interval  = 1.0 / config.DISPLAY_FPS   # FIX: cap at 30 FPS
    disp_fps_tracker = deque(maxlen=30)

    while job['running'] or not job['done']:
        if job['paused']:
            time.sleep(0.1)
            continue
        t_loop_start = time.perf_counter()
        

        frame_copy = None
        with job['frame_lock']:
            if job['latest_frame'] is not None:
                frame_copy = job['latest_frame'].copy()

        if frame_copy is None:
            time.sleep(0.01)
            continue

        with job['result_lock']:
            results = job['latest_results']

        if results:
            draw_annotations(frame_copy, results)

        frame_num    = job.get('frame_num', 0)
        total_frames = job.get('total_frames', 0)
        elapsed      = time.time() - job.get('start_time', time.time())
        inf_fps      = job.get('inf_fps', 0.0)
        disp_fps     = float(np.mean(disp_fps_tracker)) if disp_fps_tracker else 0.0
        n_students   = len(results) if results else 0
        n_suspicious = sum(1 for d in results if d['label'] == 'Suspicious') if results else 0



        _, buf = cv2.imencode('.jpg', frame_copy,
                              [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
        b64    = base64.b64encode(buf).decode()

        payload = {
            'b64'         : b64,
            'frame_num'   : frame_num,
            'total_frames': total_frames,
            'elapsed'     : elapsed,
            'inf_fps'     : round(inf_fps, 1),
            'display_fps' : round(disp_fps, 1),
            'n_students'  : n_students,
            'n_suspicious': n_suspicious,
            'activity_log': list(job['activity_log'])[-10:],
            'is_webcam'   : is_webcam,
            'paused'      : job['paused'],
            'annotations' : [
                {'track_id': d['student_id'], 'label': d['label'], 'prob': d['resnet_prob']}
                for d in (results or [])
            ],
        }

        job['frames'].append(payload)
        while len(job['frames']) > config.FRAME_BUFFER_SIZE:
            job['frames'].popleft()

        loop_elapsed = time.perf_counter() - t_loop_start
        if loop_elapsed > 0:
            disp_fps_tracker.append(1.0 / loop_elapsed)

        # FIX: sleep remainder — prevents 194 FPS bug
        sleep_time = target_interval - loop_elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    job['done'] = True

# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

def start_job_threads(job_id: str, source):
    threading.Thread(target=capture_loop,   args=(job_id, source), daemon=True).start()
    threading.Thread(target=inference_loop, args=(job_id,),        daemon=True).start()
    threading.Thread(target=display_loop,   args=(job_id,),        daemon=True).start()

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    job_id    = str(uuid.uuid4())
    save_path = UPLOAD_DIR / f"{job_id}_{file.filename}"
    with open(save_path, "wb") as f:
        f.write(await file.read())
    jobs[job_id] = make_job(is_webcam=False)
    jobs[job_id]['path'] = str(save_path)
    start_job_threads(job_id, str(save_path))
    return {"job_id": job_id}

@app.post("/start_webcam")
async def start_webcam():
    job_id       = str(uuid.uuid4())
    jobs[job_id] = make_job(is_webcam=True)
    start_job_threads(job_id, 0)
    return {"job_id": job_id}

@app.post("/stop/{job_id}")
async def stop_job(job_id: str):
    if job_id in jobs:
        jobs[job_id]['running'] = False
    return {"status": "stopped"}

@app.post("/pause/{job_id}")
async def pause_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    jobs[job_id]['paused'] = not jobs[job_id]['paused']
    return {"paused": jobs[job_id]['paused']}

@app.get("/stream/{job_id}")
async def stream_frames(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    async def event_generator():
        job         = jobs[job_id]
        empty_iters = 0
        while True:
            if job['frames']:
                payload = job['frames'].popleft()
                yield f"data: {json.dumps(payload)}\n\n"
                empty_iters = 0
            else:
                if job['done'] and not job['running']:
                    yield 'data: {"done": true}\n\n'
                    break
                empty_iters += 1
                if empty_iters > 600:
                    break
                await asyncio.sleep(0.033)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = Path("index.html")
    if html_path.exists():
        return html_path.read_text()
    return "<h1>OEPS Backend Running</h1>"

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)