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

# Создаем таблицы в БД при запуске
Base.metadata.create_all(bind=engine)

app = FastAPI(title="LLM Fine-Tuning Platform")

# Разрешаем доступ со всех IP (CORS), чтобы можно было заходить с ноута
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic схема для приема данных на старт обучения
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
    """
    Эндпоинт для загрузки файла через графический интерфейс.
    """
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

@app.get("/api/models/available")
def list_available_models():
    """
    Отдает фронтенду список доступных базовых моделей.
    """
    return [
        "meta-llama/Llama-2-7b-hf",
        "mistralai/Mistral-7B-v0.1",
        "IlyaGusev/saiga_llama3_8b"
    ]

@app.post("/api/train")
def create_training_job(request: TrainRequest, db: Session = Depends(get_db)):
    """
    Запускает Celery-задачу на основе параметров из веб-интерфейса.
    """
    if not os.path.exists(request.dataset_path):
        raise HTTPException(status_code=400, detail="Указанный датасет не найден на сервере.")

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
    """
    Стриминг метрик из Redis прямо в браузер.
    """
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

# Раздача собранного фронтенда (React/Vue). 
# Должно быть в самом конце файла, чтобы не перекрывать API роуты!
if os.path.exists("dist"):
    app.mount("/", StaticFiles(directory="dist", html=True), name="static")