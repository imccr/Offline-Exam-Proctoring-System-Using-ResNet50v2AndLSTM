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
    # ── Thread FPS Targets ────────────────────────────────────────────────────
    DISPLAY_FPS            = 30       # Thread 1 target
    INFERENCE_FPS          = 10       # Thread 2 target
    # ── Output Buffer ─────────────────────────────────────────────────────────
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
    """
    Handles YOLO → IOU Tracker → Skeleton ROI → ResNet50V2 → Decision Module.
    One instance per job to keep tracker state isolated.
    """
    def __init__(self, yolo_model, resnet_infer_fn, skeleton_gen):
        self.yolo         = yolo_model
        self.resnet_infer = resnet_infer_fn      # compiled tf.function
        self.skeleton_gen = skeleton_gen
        self.tracker      = SimpleIoUTracker(
            iou_threshold=config.IOU_THRESHOLD,
            max_lost=config.MAX_LOST_FRAMES
        )

    def process_frame(self, frame):
        # ── Step 1: YOLO Person Detection + Pose Estimation ───────────────
        rgb_frame  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results    = self.yolo(rgb_frame, conf=config.POSE_CONF_THRESHOLD, verbose=False)
        detections = []
        if results[0].keypoints is not None and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            kpts  = results[0].keypoints.data.cpu().numpy()
            for b, c, k in zip(boxes, confs, kpts):
                detections.append((b, c, k))

        # ── Step 2: IOU Tracker — Student ID Assignment ───────────────────
        matched_tracks = self.tracker.update(detections)
        if not matched_tracks:
            return []

        # ── Step 3: Skeleton ROI Extraction ──────────────────────────────
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

        # ── Step 4: ResNet50V2 Behavioral Classification (batched) ────────
        if skeleton_batch:
            preprocessed  = self.skeleton_gen.preprocess_batch(skeleton_batch)
            batch_tensor  = tf.constant(preprocessed, dtype=tf.float32)
            # Single batched GPU call via compiled tf.function (FIX #10)
            predictions   = self.resnet_infer(batch_tensor).numpy().flatten()

            for i, (track_id, bbox) in enumerate(track_info):
                prob          = float(predictions[i]) if i < len(predictions) else 0.0
                is_suspicious = 1 if prob >= config.SUSPICIOUS_THRESHOLD else 0
                self.tracker.add_vote(track_id, is_suspicious)

                # ── Step 5: Decision Module ────────────────────────────────
                label, vote_ratio = self.tracker.get_decision(track_id)
                detections_info.append({
                    'student_id' : track_id,
                    'bbox'       : bbox,
                    'label'      : label,
                    'resnet_prob': prob,
                    'vote_ratio' : vote_ratio,
                })

        # Skipped tracks (invalid skeleton) still get last known label
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

def draw_status_bar(frame, frame_num, total_frames, elapsed, inf_fps, display_fps, n_students, n_sus):
    h, w  = frame.shape[:2]
    bar_h = 36
    cv2.rectangle(frame, (0, h-bar_h), (w, h), (15,15,15), -1)
    mins, secs = divmod(int(elapsed), 60)
    progress   = (frame_num / total_frames * 100) if total_frames > 0 else 0
    status = (f'Frame:{frame_num}/{total_frames}  {progress:.1f}%  '
              f'DispFPS:{display_fps:.1f}  InfFPS:{inf_fps:.1f}  '
              f'Time:{mins:02d}:{secs:02d}  '
              f'Students:{n_students}  Suspicious:{n_sus}')
    cv2.putText(frame, status, (8, h-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,220,0), 1, cv2.LINE_AA)
    return frame

# ── Global Model Loading (once at startup) ────────────────────────────────────
print("Loading models...")
device       = 'cuda' if torch.cuda.is_available() else 'cpu'
yolo_model   = YOLO('yolo11s-pose.pt').to(device)
resnet_model = load_model(RESNET_MODEL_PATH, compile=False)
skeleton_gen = SkeletonGenerator()

# ── FIX #10: Compile ResNet as tf.function for GPU graph optimization ─────────
@tf.function(
    input_signature=[tf.TensorSpec(shape=[None, 224, 224, 3], dtype=tf.float32)],
    reduce_retracing=True
)
def resnet_infer(x):
    return resnet_model(x, training=False)

# ── FIX #9: Warmup with multiple batch sizes to pre-compile TF graph ──────────
print("Warming up ResNet (pre-compiling TF graph for all batch sizes)...")
for bs in [1, 2, 4, 8]:
    dummy = tf.random.normal([bs, 224, 224, 3])
    _     = resnet_infer(dummy)
print(f"Models ready. Device: {device}")

# ── Job State ─────────────────────────────────────────────────────────────────
# FIX #7: Per-job state dictionary to support multiple concurrent videos
jobs: dict = {}
# Structure per job:
# {
#   'frames'      : deque,        # encoded frames ready to stream
#   'done'        : bool,         # video finished
#   'running'     : bool,         # threads active (FIX #8)
#   'frame_lock'  : Lock,         # protects latest_frame
#   'result_lock' : Lock,         # protects latest_results
#   'new_frame'   : Event,        # signals inference thread (FIX #3)
#   'latest_frame': None|ndarray,
#   'latest_results': None|list,
#   'total_frames': int,
#   'frame_num'   : int,
#   'start_time'  : float,
#   'activity_log': list,
#   'inf_fps'     : float,
#   'display_fps' : float,
# }

# ── Thread 1: Capture Loop (~30 FPS) ─────────────────────────────────────────
def capture_loop(job_id: str, video_path: str):
    """
    Reads frames from video at target DISPLAY_FPS.
    Writes to shared latest_frame.
    Never waits for AI inference.
    FIX #1: Sleep only the REMAINDER after cap.read() time.
    """
    job           = jobs[job_id]
    cap           = cv2.VideoCapture(video_path)
    total_frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    job['total_frames'] = total_frames

    target_interval = 1.0 / config.DISPLAY_FPS
    frame_num       = 0

    while job['running']:
        t_read_start = time.perf_counter()
        ret, frame   = cap.read()

        if not ret:
            # Video ended
            job['done'] = True
            break

        frame_num += 1

        # Write latest frame to shared memory
        with job['frame_lock']:
            job['latest_frame'] = frame.copy()
            job['frame_num']    = frame_num

        # Signal inference thread that a new frame is available (FIX #3)
        job['new_frame'].set()

        # FIX #1: Sleep only remainder to hit target FPS
        elapsed_read = time.perf_counter() - t_read_start
        sleep_time   = target_interval - elapsed_read
        if sleep_time > 0:
            time.sleep(sleep_time)

    cap.release()
    job['running'] = False

# ── Thread 2: Inference Loop (~8-10 FPS) ─────────────────────────────────────
def inference_loop(job_id: str):
    """
    Waits for new frame signal, runs full AI pipeline, stores results.
    FIX #2: Uses Event.wait(timeout) to avoid busy-waiting.
    FIX #3: Processes only when new frame is signaled.
    """
    job              = jobs[job_id]
    pipe             = AIPipeline(yolo_model, resnet_infer, skeleton_gen)
    target_interval  = 1.0 / config.INFERENCE_FPS
    inf_fps_tracker  = deque(maxlen=20)

    while job['running'] or not job['done']:
        # FIX #2 & #3: Wait for new frame signal with timeout
        # This avoids busy-waiting and ensures we process fresh frames
        got_new = job['new_frame'].wait(timeout=1.0)
        if not got_new:
            if job['done']:
                break
            continue

        # Clear the event so we wait for the next new frame
        job['new_frame'].clear()

        # Grab latest frame copy
        frame_copy = None
        with job['frame_lock']:
            if job['latest_frame'] is not None:
                frame_copy = job['latest_frame'].copy()

        if frame_copy is None:
            continue

        # Run full AI pipeline
        t0              = time.perf_counter()
        detections_info = pipe.process_frame(frame_copy)
        elapsed         = time.perf_counter() - t0

        # Update inference FPS
        if elapsed > 0:
            inf_fps_tracker.append(1.0 / elapsed)
        job['inf_fps'] = float(np.mean(inf_fps_tracker)) if inf_fps_tracker else 0.0

        # Update activity log for suspicious detections
        start_time = job.get('start_time', time.time())
        mins, secs = divmod(int(time.time() - start_time), 60)
        for d in detections_info:
            if d['label'] == 'Suspicious' and d['resnet_prob'] >= 0:
                job['activity_log'].append({
                    'time'       : f'{mins:02d}:{secs:02d}',
                    'track_id'   : d['student_id'],
                    'resnet_prob': d['resnet_prob'],
                })
                # Keep log bounded to last 100 entries
                if len(job['activity_log']) > 100:
                    job['activity_log'].pop(0)

        # Write results to shared memory
        with job['result_lock']:
            job['latest_results'] = detections_info

        # FIX #2: Enforce upper bound — sleep remainder to stay at INFERENCE_FPS
        # This prevents inference from running uncapped and starving display
        sleep_time = target_interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

# ── Thread 3: Display + Encode Loop (~30 FPS) ─────────────────────────────────
def display_loop(job_id: str):
    """
    Reads latest_frame + latest_results.
    Draws annotations without waiting for AI.
    Encodes JPEG and pushes to output frame buffer.
    """
    job              = jobs[job_id]
    target_interval  = 1.0 / config.DISPLAY_FPS
    disp_fps_tracker = deque(maxlen=30)

    while job['running'] or not job['done']:
        t_loop_start = time.perf_counter()

        # Read latest frame
        frame_copy = None
        with job['frame_lock']:
            if job['latest_frame'] is not None:
                frame_copy = job['latest_frame'].copy()

        if frame_copy is None:
            time.sleep(0.01)
            continue

        # Read latest AI results (may be None if inference hasn't run yet)
        with job['result_lock']:
            results = job['latest_results']

        # Draw annotations from last known AI results
        if results:
            draw_annotations(frame_copy, results)

        # Gather stats
        frame_num    = job.get('frame_num', 0)
        total_frames = job.get('total_frames', 0)
        elapsed      = time.time() - job.get('start_time', time.time())
        inf_fps      = job.get('inf_fps', 0.0)
        disp_fps     = float(np.mean(disp_fps_tracker)) if disp_fps_tracker else 0.0
        n_students   = len(results) if results else 0
        n_suspicious = sum(1 for d in results if d['label'] == 'Suspicious') if results else 0

        # Draw status bar on frame (FIX #6)
        draw_status_bar(frame_copy, frame_num, total_frames,
                        elapsed, inf_fps, disp_fps,
                        n_students, n_suspicious)

        # Encode frame to JPEG
        _, buf = cv2.imencode('.jpg', frame_copy,
                              [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
        b64    = base64.b64encode(buf).decode()

        # Build full payload (FIX #5: all fields frontend expects)
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
            'annotations' : [
                {
                    'track_id': d['student_id'],
                    'label'   : d['label'],
                    'prob'    : d['resnet_prob'],
                }
                for d in (results or [])
            ],
        }

        job['frames'].append(payload)

        # Keep buffer bounded (FIX #7 side effect)
        while len(job['frames']) > config.FRAME_BUFFER_SIZE:
            job['frames'].popleft()

        # Update display FPS tracker
        loop_elapsed = time.perf_counter() - t_loop_start
        if loop_elapsed > 0:
            disp_fps_tracker.append(1.0 / loop_elapsed)

        # Sleep remainder to target DISPLAY_FPS
        sleep_time = target_interval - loop_elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    # Signal stream that video is done
    job['done'] = True

# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# ── FIX #4: /upload endpoint restored ────────────────────────────────────────
@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    job_id    = str(uuid.uuid4())
    save_path = UPLOAD_DIR / f"{job_id}_{file.filename}"

    with open(save_path, "wb") as f:
        f.write(await file.read())

    # FIX #7 + #8: Per-job isolated state, fresh running=True each time
    jobs[job_id] = {
        'frames'         : deque(),
        'done'           : False,
        'running'        : True,          # FIX #8: per-job flag, not global
        'frame_lock'     : threading.Lock(),
        'result_lock'    : threading.Lock(),
        'new_frame'      : threading.Event(),   # FIX #3
        'latest_frame'   : None,
        'latest_results' : None,
        'total_frames'   : 0,
        'frame_num'      : 0,
        'start_time'     : time.time(),
        'activity_log'   : [],
        'inf_fps'        : 0.0,
        'display_fps'    : 0.0,
        'path'           : str(save_path),
    }

    # Start all 3 threads per job
    threading.Thread(
        target=capture_loop, args=(job_id, str(save_path)), daemon=True
    ).start()
    threading.Thread(
        target=inference_loop, args=(job_id,), daemon=True
    ).start()
    threading.Thread(
        target=display_loop, args=(job_id,), daemon=True
    ).start()

    return {"job_id": job_id}

# ── FIX #4: Stream endpoint restored to /stream/{job_id} ─────────────────────
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
                if empty_iters > 600:   # 30s timeout
                    break
                await asyncio.sleep(0.033)   # poll at ~30fps

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = Path("index.html")
    if html_path.exists():
        return html_path.read_text()
    return "<h1>OEPS Backend Running</h1>"

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)