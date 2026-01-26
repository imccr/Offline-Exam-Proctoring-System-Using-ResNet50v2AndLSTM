import numpy as np
import tensorflow as tf
from app.core.config import settings
from app.services.preprocessor import extract_frames

# Load model ONCE at startup
# model = tf.keras.models.load_model(settings.MODEL_PATH)

def predict_video(video_path: str):
    # frames = extract_frames(
    #     video_path,
    #     settings.FRAME_STRIDE,
    #     settings.FRAME_SIZE
    # )

    # if len(frames) == 0:
    #     return {
    #         "label": "normal",
    #         "total_frames": 0,
    #         "suspicious_frames": 0,
    #         "suspicious_ratio": 0.0
    #     }

    # # Model outputs probability [0,1]
    # probs = model.predict(frames, verbose=0).reshape(-1)

    # suspicious_frames = int((probs >= 0.5).sum())
    # ratio = suspicious_frames / len(probs)

    # label = "suspicious" if ratio >= settings.SUSPICIOUS_THRESHOLD else "normal"

    return {}
        

        # "label": label,
        # "total_frames": int(len(probs)),
        # "suspicious_frames": suspicious_frames,
        # "suspicious_ratio": round(float(ratio), 3)
    
