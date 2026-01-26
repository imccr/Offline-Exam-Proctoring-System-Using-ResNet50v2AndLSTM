from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(title="Offline Exam Proctoring API")

app.include_router(router)

@app.get("/")
def health():
    return {"status": "API running"}
