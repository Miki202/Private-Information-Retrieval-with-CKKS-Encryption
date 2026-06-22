import os
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