from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os

# URL за връзка с базата данни
# Ако си сложил друга парола при инсталацията, промени тук
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres123@localhost:5432/vehicle_storage"
    # Формат: postgresql://username:password@host:port/database_name
)

# Създаване на engine за връзка с DB
engine = create_engine(DATABASE_URL, echo=False)  # echo=True за да виждаш SQL queries
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

def get_db():
    """
    Функция за получаване на database session
    Използва се във всички операции с базата
    """
    db = SessionLocal()
    try:
        return db
    except:
        db.close()
        raise

def test_connection():
    """
    Тества връзката с базата данни
    """
    try:
        db = get_db()
        db.execute("SELECT 1")  # Проста проверка
        db.close()
        print("Връзката с базата данни е успешна")
        return True
    except Exception as e:
        print(f"Връзката с базата данни е неуспешна: {e}")
        return Falseimport os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL липсва в .env")

engine = create_engine(DATABASE_URL, echo=False)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False
)

Base = declarative_base()

def get_db():
    """Създава и връща DB session"""
    return SessionLocal()

def test_connection():
    """Тества връзката към PostgreSQL"""
    db = get_db()
    try:
        db.execute(text("SELECT 1"))
        print("Успешно свързване")
        return True
    except Exception as e:
        print(f"Неуспешна връзка: {e}")
        return False
    finally:
        db.close()