"""
SQLAlchemy модел за TRUE PIR с plaintext за UI
"""
from sqlalchemy import Column, Integer, String, DateTime, LargeBinary
from datetime import datetime
from .connection import Base


class Vehicle(Base):
    """
    Модел за превозното средство

    Архитектура:
    - Криптирани данни за PIR (encrypted_embedding, encrypted_metadata)
    - Plaintext метаданни за UI визуализация и Stage 1 bucketization
    """

    __tablename__ = "vehicles"

    # Идентификатори
    id = Column(Integer, primary_key=True, index=True)
    vehicle_uuid = Column(String, unique=True, nullable=False)

    # Plaintext метаданни (за UI + Stage 1 bucket filtering)
    license_plate = Column(String, index=True)
    color = Column(String, index=True)
    body_type = Column(String, index=True)
    image_path = Column(String)

    # Криптирани данни (за Stage 2 PIR търсене)
    encrypted_embedding = Column(LargeBinary, nullable=False)
    encrypted_metadata = Column(LargeBinary, nullable=False)

    # Времеви маркери
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Vehicle {self.license_plate} (TRUE_PIR)>"

    def to_dict(self):
        """Конвертира в речник"""
        return {
            "id": self.id,
            "uuid": self.vehicle_uuid,
            "license_plate": self.license_plate,
            "color": self.color,
            "body_type": self.body_type,
            "image_path": self.image_path,
            "created_at": self.created_at,
        }

    def get_storage_size(self) -> int:
        """Изчислява размера на криптираните данни в байтове"""
        size = 0
        if self.encrypted_embedding:
            size += len(self.encrypted_embedding)
        if self.encrypted_metadata:
            size += len(self.encrypted_metadata)
        return size
