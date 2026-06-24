from sqlalchemy import Column, Integer, String, DateTime, LargeBinary
from datetime import datetime
from storage.database.connection import Base


class Vehicle(Base):
    """
    Модел за превозно средство в PURE PIR система

    Архитектура:
    - Всичко е криптирано (CKKS)
    - Няма plaintext полета
    - Серверът не може да види съдържанието
    """

    __tablename__ = "vehicles"

    # Идентификатори
    id = Column(Integer, primary_key=True, index=True)
    vehicle_uuid = Column(String, unique=True, nullable=False)

    # Криптирани данни
    encrypted_embedding = Column(LargeBinary, nullable=False)
    encrypted_metadata = Column(LargeBinary, nullable=False)

    # Времеви маркери
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        """
        Текстово представяне на обекта (само за debug)
        """
        return f"<Vehicle id={self.id} PIR_ENCRYPTED>"

    def to_dict(self):
        """
        Конвертира обекта в речник.

        Забележка:
        Метаданните са криптирани и трябва да се декриптират клиентски.
        """
        return {
            "id": self.id,
            "uuid": self.vehicle_uuid,
            "is_encrypted": True,
            "has_encrypted_data": True,
            "encrypted_embedding_size": len(self.encrypted_embedding) if self.encrypted_embedding else 0,
            "encrypted_metadata_size": len(self.encrypted_metadata) if self.encrypted_metadata else 0,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def get_storage_size(self) -> int:
        """
        Изчислява общия размер на криптираните данни в байтове
        """
        size = 0

        if self.encrypted_embedding:
            size += len(self.encrypted_embedding)

        if self.encrypted_metadata:
            size += len(self.encrypted_metadata)

        return size