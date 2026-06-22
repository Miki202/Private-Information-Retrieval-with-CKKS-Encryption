"""
SQLAlchemy models для TRUE PIR
"""
from sqlalchemy import Column, Integer, String, DateTime, Boolean, LargeBinary
from datetime import datetime
from .connection import Base

class Vehicle(Base):
    """
    Vehicle model with TRUE PIR support
    
    Modes:
    - Plain (is_encrypted=False): Uses FAISS for fast search
    - Encrypted (is_encrypted=True): Uses TRUE PIR with homomorphic operations
    """
    __tablename__ = "vehicles"
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True)
    vehicle_uuid = Column(String, unique=True, nullable=False)
    
    # Метаданни (visible to server)
    license_plate = Column(String, index=True)
    color = Column(String, index=True)
    body_type = Column(String, index=True)
    image_path = Column(String)
    
    # Mode flag
    is_encrypted = Column(Boolean, default=False, index=True)
    
    # Plain mode (FAISS)
    faiss_id = Column(Integer, index=True)
    
    # TRUE PIR mode (homomorphic)
    encrypted_embedding = Column(LargeBinary)  # ~100KB
    encryption_context = Column(LargeBinary)   # ~50KB
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        mode = "TRUE-PIR" if self.is_encrypted else "PLAIN"
        return f"<Vehicle {self.license_plate} ({mode})>"
    
    def to_dict(self):
        """Конвертира в dictionary (без encrypted data)"""
        return {
            "id": self.id,
            "uuid": self.vehicle_uuid,
            "license_plate": self.license_plate,
            "color": self.color,
            "body_type": self.body_type,
            "image_path": self.image_path,
            "is_encrypted": self.is_encrypted,
            "faiss_id": self.faiss_id,
            "has_encrypted_data": self.encrypted_embedding is not None,
            "encrypted_size": len(self.encrypted_embedding) if self.encrypted_embedding else 0,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
    
    def get_storage_size(self) -> int:
        """Returns total storage size in bytes"""
        size = 0
        if self.encrypted_embedding:
            size += len(self.encrypted_embedding)
        if self.encryption_context:
            size += len(self.encryption_context)
        return size