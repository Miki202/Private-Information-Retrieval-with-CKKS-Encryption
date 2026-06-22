"""
SQLAlchemy models for TRUE PIR with encrypted metadata
"""
from sqlalchemy import Column, Integer, String, DateTime, Boolean, LargeBinary
from datetime import datetime
from .connection import Base

class Vehicle(Base):
    """
    Vehicle model with TRUE PIR support
    
    Modes:
    - Plain (is_encrypted=False): 
        * Uses FAISS for fast search
        * Metadata visible (license_plate, color, body_type)
    
    - Encrypted (is_encrypted=True): 
        * Uses TRUE PIR with homomorphic operations
        * Everything encrypted (embeddings + metadata)
        * Server sees nothing
    """
    __tablename__ = "vehicles"
    
    id = Column(Integer, primary_key=True, index=True)
    vehicle_uuid = Column(String, unique=True, nullable=False)
    
    # Mode flag
    is_encrypted = Column(Boolean, default=False, index=True)
    
    # Plain mode (FAISS) - metadata visible
    faiss_id = Column(Integer, index=True)
    license_plate = Column(String, index=True)
    color = Column(String, index=True)
    body_type = Column(String, index=True)
    image_path = Column(String)
    
    # TRUE PIR mode - everything encrypted
    encrypted_embedding = Column(LargeBinary)   # ~100KB
    encrypted_metadata = Column(LargeBinary)    # ~50KB
    encryption_context = Column(LargeBinary)    # ~50KB
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        mode = "TRUE-PIR" if self.is_encrypted else "PLAIN"
        if self.is_encrypted:
            return f"<Vehicle [ENCRYPTED] ({mode})>"
        else:
            return f"<Vehicle {self.license_plate} ({mode})>"
    
    def to_dict(self):
        """Convert to dictionary (decrypted metadata must be provided separately)"""
        if self.is_encrypted:
            return {
                "id": self.id,
                "uuid": self.vehicle_uuid,
                "is_encrypted": True,
                "has_encrypted_data": True,
                "encrypted_size": self.get_storage_size(),
                "created_at": self.created_at,
                "updated_at": self.updated_at
            }
        else:
            return {
                "id": self.id,
                "uuid": self.vehicle_uuid,
                "license_plate": self.license_plate,
                "color": self.color,
                "body_type": self.body_type,
                "image_path": self.image_path,
                "is_encrypted": False,
                "faiss_id": self.faiss_id,
                "created_at": self.created_at,
                "updated_at": self.updated_at
            }
    
    def get_storage_size(self) -> int:
        """Returns total storage size in bytes"""
        size = 0
        if self.encrypted_embedding:
            size += len(self.encrypted_embedding)
        if self.encrypted_metadata:
            size += len(self.encrypted_metadata)
        if self.encryption_context:
            size += len(self.encryption_context)
        return size