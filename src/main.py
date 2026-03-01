import os
import json
from fastapi import FastAPI, Depends, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
import redis.asyncio as aioredis
from pydantic import BaseModel

from .database import engine, Base, SessionLocal
from .models import TrainingJob
from .tasks import run_training_task
from .dataset_parser import DatasetParser

Base.metadata.create_all(bind=engine)

app = FastAPI(title="LLM Fine-Tuning Platform")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TrainRequest(BaseModel):
    dataset_path: str
    params: dict

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/api/dataset/upload")
async def upload_dataset(file: UploadFile = File(...)):
    os.makedirs("raw_data", exist_ok=True)
    file_path = f"raw_data/{file.filename}"
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    parser = DatasetParser()
    is_valid, result = parser.load_and_validate(file_path)
    if not is_valid:
        os.remove(file_path)
        raise HTTPException(status_code=400, detail=result)
    return {"message": "Файл прошел валидацию и сохранен", "file_path": file_path}

@app.get("/api/model/check")
def check_model_path(path: str):
    """Проверяет, есть ли такая папка на сервере или это HF ID"""
    if os.path.exists(path):
        return {"status": "ok", "type": "local_directory"}
    if "/" in path and len(path.split("/")) == 2:
        return {"status": "ok", "type": "huggingface_id"}
    raise HTTPException(status_code=404, detail="Путь не найден на сервере и не похож на HF ID")

@app.post("/api/train")
def create_training_job(request: TrainRequest, db: Session = Depends(get_db)):
    if not os.path.exists(request.dataset_path):
        raise HTTPException(status_code=400, detail="Датасет не найден")

    job = TrainingJob(model_type="local", params=request.params)
    db.add(job)
    db.commit()
    db.refresh(job)
    
    parser = DatasetParser()
    _, valid_data = parser.load_and_validate(request.dataset_path)
    processed_path = f"processed_data/dataset_{job.id}.jsonl"
    parser.save_standardized(valid_data, processed_path)
    
    run_training_task.delay(job.id)
    return {"task_id": job.id, "status": job.status}

@app.websocket("/ws/metrics/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: int):
    await websocket.accept()
    redis_client = await aioredis.from_url("redis://localhost:6379/0", decode_responses=True)
    pubsub = redis_client.pubsub()
    channel_name = f"training_metrics_{task_id}"
    await pubsub.subscribe(channel_name)
    try:
        async for message in pubsub.listen():
            if message["type"] == "message":
                await websocket.send_text(message["data"])
    except WebSocketDisconnect:
        pass
    finally:
        await pubsub.unsubscribe(channel_name)
        await redis_client.aclose()

if os.path.exists("dist"):
    app.mount("/", StaticFiles(directory="dist", html=True), name="static")