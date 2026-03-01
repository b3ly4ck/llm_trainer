from celery import Celery

# Создаем экземпляр Celery
# broker: куда FastAPI отправляет задачи (Redis 0-я база)
# backend: где Celery хранит результаты выполнения (Redis 1-я база)
celery_app = Celery(
    "llm_tasks",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/1",
    include=["src.tasks"]  # Указываем, где искать определения задач
)

# Опциональные настройки для стабильности
celery_app.conf.update(
    task_track_started=True,
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)