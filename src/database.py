from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# Путь к файлу базы данных SQLite
DATABASE_URL = "sqlite:///./llm_platform.db" 

# Создаем движок. check_same_thread=False нужен только для SQLite
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

# Настраиваем фабрику сессий для взаимодействия с БД
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Базовый класс для всех моделей (от него наследуется TrainingJob в models.py)
Base = declarative_base()