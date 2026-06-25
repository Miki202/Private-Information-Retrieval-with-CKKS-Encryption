"""
SQLAlchemy модел за TRUE PIR с plaintext за UI
"""
from sqlalchemy import Column, Integer, String, DateTime, LargeBinary
from datetime import datetime
from storage.database.connection import Base
from storage.database.connection import Base


class Vehicle(Base):
    """
    Модел за превозното средство
    
    Архитектура:
    - Криптирани данни за PIR
    - Plaintext метаданни за UI
    - Криптирани данни за PIR
    - Plaintext метаданни за UI
    """
    
    
    __tablename__ = "vehicles"
    
    
    # Идентификатори
    id = Column(Integer, primary_key=True, index=True)
    vehicle_uuid = Column(String, unique=True, nullable=False)
    
    # Plaintext метаданни (за UI)
    license_plate = Column(String, index=True)
    color = Column(String, index=True)
    body_type = Column(String, index=True)
    image_path = Column(String)
    
    # Криптирани данни (за PIR)
    
    # Plaintext метаданни (за UI)
    license_plate = Column(String, index=True)
    color = Column(String, index=True)
    body_type = Column(String, index=True)
    image_path = Column(String)
    
    # Криптирани данни (за PIR)
    encrypted_embedding = Column(LargeBinary, nullable=False)
    encrypted_metadata = Column(LargeBinary, nullable=False)
    
    
    # Времеви маркери
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime
.utcnow)
    
    updated_at = Column(DateTime, default=datetime
.utcnow)
    
    def __repr__(self):
        return f"<Vehicle {self.license_plate} (TRUE_PIR)>"
    
        return f"<Vehicle {self.license_plate} (TRUE_PIR)>"
    
    def to_dict(self):
        """Конвертира в речник"""
        """Конвертира в речник"""
        return {
            "id": self.id,
            "uuid": self.vehicle_uuid,
            "license_plate": self.license_plate,
            "color": self.color,
            "body_type": self.body_type,
            "image_path": self.image_path,
            "license_plate": self.license_plate,
            "color": self.color,
            "body_type": self.body_type,
            "image_path": self.image_path,
            "created_at": self.created_at,
        }
        }