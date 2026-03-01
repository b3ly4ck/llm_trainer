from sqlalchemy import Column, Integer, String, JSON
from .database import Base

class TrainingJob(Base):
    __tablename__ = "training_jobs"
    
    id = Column(Integer, primary_key=True, index=True)
    status = Column(String, default="Pending") # Статусы: Pending, Running, Completed, Failed
    model_type = Column(String, default="local") # Оставили только локальное обучение
    params = Column(JSON) # Сохраненные гиперпараметры (epochs, batch_size, lr и т.д.)
    metrics = Column(JSON, default={}) # Финальные результаты или текст ошибки при краше