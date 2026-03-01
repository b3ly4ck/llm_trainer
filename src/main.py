from fastapi import FastAPI, Depends, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from sqlalchemy.orm import Session
import redis.asyncio as aioredis
import json
import os

from .database import engine, Base, SessionLocal
from .models import TrainingJob
from .tasks import run_training_task
from .dataset_parser import DatasetParser

# Создаем таблицы в БД при запуске
Base.metadata.create_all(bind=engine)

app = FastAPI(title="LLM Fine-Tuning Platform")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/api/dataset/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """
    Загрузка, валидация и сохранение сырого датасета на сервере.
    """
    os.makedirs("raw_data", exist_ok=True)
    raw_path = f"raw_data/{file.filename}"
    
    with open(raw_path, "wb") as f:
        content = await file.read()
        f.write(content)
        
    parser = DatasetParser()
    is_valid, result = parser.load_and_validate(raw_path)
    
    if not is_valid:
        os.remove(raw_path)
        raise HTTPException(status_code=400, detail=result)
        
    return {"message": "Датасет успешно прошел валидацию", "file_path": raw_path}

@app.post("/api/train")
def create_training_job(params: dict, dataset_path: str, db: Session = Depends(get_db)):
    """
    Создание задачи на дообучение. Конвертирует сырой датасет в JSONL и кидает задачу в Celery.
    """
    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=400, detail="Указанный датасет не найден.")

    job = TrainingJob(model_type="local", params=params)
    db.add(job)
    db.commit()
    db.refresh(job)
    
    parser = DatasetParser()
    _, valid_data = parser.load_and_validate(dataset_path)
    processed_path = f"processed_data/dataset_{job.id}.jsonl"
    parser.save_standardized(valid_data, processed_path)
    
    run_training_task.delay(job.id)
    
    return {"task_id": job.id, "status": job.status}

@app.get("/api/status/{task_id}")
def get_status(task_id: int, db: Session = Depends(get_db)):
    """
    Проверка статического статуса задачи в БД.
    """
    job = db.query(TrainingJob).filter(TrainingJob.id == task_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Задача не найдена")
        
    return {"task_id": job.id, "status": job.status, "metrics": job.metrics}

@app.websocket("/ws/metrics/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: int):
    """
    WebSocket-эндпоинт для стриминга графиков Loss и статусов из Redis напрямую на клиент.
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
                
                # Закрываем сокет, если процесс завершился или упал
                data = json.loads(message["data"])
                if data.get("status") in ["Completed", "Failed"]:
                    break
    except WebSocketDisconnect:
        pass
    finally:
        await pubsub.unsubscribe(channel_name)
        await redis_client.aclose()
        # Проверка, чтобы не закрыть уже закрытый сокет
        try:
            await websocket.close()
        except RuntimeError:
            pass