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
        return False