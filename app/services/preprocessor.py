import cv2
import numpy as np

def extract_frames(video_path: str, stride: int, size: int):
    cap = cv2.VideoCapture(video_path)
    frames = []
    idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if idx % stride == 0:
            frame = cv2.resize(frame, (size, size))
            frame = frame.astype("float32") / 255.0
            frames.append(frame)

        idx += 1

    cap.release()
    return np.array(frames)
