import os
import uuid
from fastapi import APIRouter, UploadFile, File
from app.core.config import settings
from app.services.model_manager import predict_video

router = APIRouter()

os.makedirs(settings.VIDEO_DIR, exist_ok=True)

@router.post("/analyze-video")
async def analyze_video(
    student_id: str,
    video: UploadFile = File(...)
):
    video_name = f"{student_id}_{uuid.uuid4()}.mp4"
    video_path = os.path.join(settings.VIDEO_DIR, video_name)

    with open(video_path, "wb") as f:
        f.write(await video.read())

    result = predict_video(video_path)

    return {
        "student_id": student_id,
        **result
    }
