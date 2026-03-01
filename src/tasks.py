import redis
import json
from .celery_app import celery_app
from .database import SessionLocal
from .models import TrainingJob

redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

@celery_app.task(bind=True, name="run_training_process")
def run_training_task(self, job_id: int):
    db = SessionLocal()
    job = db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    
    if not job:
        return "Job not found"

    job.status = "Running"
    db.commit()

    try:
        dataset_path = f"processed_data/dataset_{job_id}.jsonl"
        
        # Читаем метод обучения из гиперпараметров (по умолчанию lora)
        training_method = job.params.get("method", "lora")
        
        if training_method == "full":
            from .workers.full_trainer import train_full
            train_full(
                job_id=job_id, 
                params=job.params, 
                redis_client=redis_client,
                dataset_path=dataset_path
            )
        else:
            from .workers.local_trainer import train_local
            train_local(
                job_id=job_id, 
                params=job.params, 
                redis_client=redis_client,
                dataset_path=dataset_path
            )
        
        job.status = "Completed"
        
    except Exception as e:
        job.status = "Failed"
        error_msg = str(e)
        job.metrics = {"error": error_msg}
        
        redis_client.publish(f"training_metrics_{job_id}", json.dumps({
            "status": "Failed",
            "error": error_msg
        }))
        
    finally:
        db.commit()
        db.close()