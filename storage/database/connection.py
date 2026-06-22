"""
PostgreSQL database connection
"""
from sqlalchemy import create_engine, text  # ← Добави text import
from sqlalchemy.orm import sessionmaker, declarative_base
import os

# Database URL
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres123@localhost:5432/vehicle_storage"
)

print(f"🔗 Connecting to: {DATABASE_URL.split('@')[1]}")

# Create engine
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

def get_db():
    """
    Връща database session
    """
    db = SessionLocal()
    try:
        return db
    except:
        db.close()
        raise

def test_connection():
    """
    Тества връзката
    """
    try:
        db = get_db()
        db.execute(text("SELECT 1"))  # ← Fix: wrap with text()
        db.close()
        print("✓ PostgreSQL connection успешна")
        return True
    except Exception as e:
        print(f"✗ PostgreSQL connection неуспешна: {e}")
        return False