"""
SQLAlchemy models за PostgreSQL
"""
from sqlalchemy import Column, Integer, String, DateTime, Boolean
from datetime import datetime
from .connection import Base

class Vehicle(Base):
    """
    Метаданни за превозни средства в PostgreSQL
    """
    __tablename__ = "vehicles"
    
    # Колони
    id = Column(Integer, primary_key=True, index=True)
    vehicle_uuid = Column(String, unique=True, nullable=False)
    
    # Метаданни
    license_plate = Column(String, index=True)
    color = Column(String, index=True)
    body_type = Column(String, index=True)
    image_path = Column(String)
    
    # Encryption & FAISS
    is_encrypted = Column(Boolean, default=False, index=True)
    faiss_id = Column(Integer, index=True)  # ID в FAISS index
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<Vehicle {self.license_plate} (FAISS:{self.faiss_id})>"
    
    def to_dict(self):
        """Конвертира в dictionary"""
        return {
            "id": self.id,
            "uuid": self.vehicle_uuid,
            "license_plate": self.license_plate,
            "color": self.color,
            "body_type": self.body_type,
            "image_path": self.image_path,
            "is_encrypted": self.is_encrypted,
            "faiss_id": self.faiss_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }