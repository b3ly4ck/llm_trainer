import redis
import json
from .celery_app import celery_app
from .database import SessionLocal
from .models import TrainingJob
from .workers.local_trainer import train_local

# Подключение к Redis для отправки статусов "Ошибка" или "Завершено" напрямую в БД и WebSockets
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

@celery_app.task(bind=True, name="run_training_process")
def run_training_task(self, job_id: int):
    """
    Основная Celery-задача для запуска локального обучения.
    """
    db = SessionLocal()
    job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    
    if not job:
        return "Job not found"

    # Обновляем статус в БД на "В процессе"
    job.status = "Running"
    db.commit()

    try:
        # Путь к обработанному датасету (формируется DatasetParser-ом)
        dataset_path = f"processed_data/dataset_{job_id}.jsonl"
        
        # Запуск универсального локального тренера
        # Мы передаем redis_client, чтобы тренер мог пушить метрики Loss в реальном времени
        train_local(
            job_id=job_id, 
            params=job.params, 
            redis_client=redis_client,
            dataset_path=dataset_path
        )
        
        # Если выполнение дошло сюда — всё успешно
        job.status = "Completed"
        
    except Exception as e:
        # Ловим любые ошибки (OOM, ошибки CUDA, отсутствие файлов)
        job.status = "Failed"
        error_msg = str(e)
        job.metrics = {"error": error_msg}
        
        # Дублируем ошибку в WebSocket канал, чтобы пользователь сразу увидел "красный" статус
        redis_client.publish(f"training_metrics_{job_id}", json.dumps({
            "status": "Failed",
            "error": error_msg
        }))
        
    finally:
        db.commit()
        db.close()