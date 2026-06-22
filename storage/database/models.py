from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, LargeBinary, Boolean
from pgvector.sqlalchemy import Vector
from datetime import datetime
from .connection import Base

class Vehicle(Base):
    """
    SQLAlchemy модел за таблицата vehicles
    Съхранява метаданните за превозните средства
    """
    __tablename__ = "vehicles"
    
    # Колони
    id = Column(Integer, primary_key=True, index=True)
    vehicle_uuid = Column(String, unique=True, nullable=False)  # Уникален идентификатор
    license_plate = Column(String, index=True)  # Номер
    color = Column(String, index=True)          # Цвят
    body_type = Column(String, index=True)      # Тип каросерия
    image_path = Column(String)                 # Път до снимка
    is_encrypted = Column(Boolean, default=False, index=True)  # Дали е криптирано
    created_at = Column(DateTime, default=datetime.utcnow)     # Кога е създадено
    updated_at = Column(DateTime, default=datetime.utcnow)     # Последна промяна
    
    def __repr__(self):
        return f"<Vehicle {self.license_plate} ({self.color} {self.body_type})>"

class VehicleVector(Base):
    """
    SQLAlchemy модел за таблицата vehicle_vectors
    Съхранява векторните embeddings (обикновени и криптирани)
    """
    __tablename__ = "vehicle_vectors"
    
    # Колони
    id = Column(Integer, primary_key=True)
    vehicle_id = Column(Integer, ForeignKey("vehicles.id"), unique=True)  # Връзка към vehicles таблицата
    
    # Обикновен embedding (256 float числа)
    embedding = Column(Vector(256))  # pgvector тип
    
    # Криптиран embedding (binary данни)
    encrypted_embedding = Column(LargeBinary)
    encryption_context = Column(LargeBinary)  # CKKS контекст
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<VehicleVector vehicle_id={self.vehicle_id}>"