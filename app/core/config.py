from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    MODEL_PATH: str = "models_data/model.keras"
    VIDEO_DIR: str = "temp_videos"
    FRAME_STRIDE: int = 5              # take every 5th frame
    FRAME_SIZE: int = 224
    SUSPICIOUS_THRESHOLD: float = 0.5  # % frames suspicious

settings = Settings()
