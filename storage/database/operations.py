from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import List, Optional, Tuple, Dict
import uuid
import numpy as np
from datetime import datetime

from .models import Vehicle, VehicleVector
from .connection import get_db

# ============================================
# INSERT ОПЕРАЦИИ (Вмъкване на данни)
# ============================================

def insert_vehicle(
    license_plate: Optional[str] = None,
    color: Optional[str] = None,
    body_type: Optional[str] = None,
    embedding: Optional[np.ndarray] = None,
    encrypted_embedding: Optional[bytes] = None,
    encryption_context: Optional[bytes] = None,
    image_path: Optional[str] = None,
) -> Tuple[str, int]:
    """
    Вмъква превозно средство с метаданни и embedding (обикновен или криптиран)
    
    Args:
        license_plate: Номер на колата
        color: Цвят
        body_type: Тип каросерия (sedan, suv, etc.)
        embedding: Обикновен 256-мерен embedding (numpy array или list)
        encrypted_embedding: Криптиран embedding (bytes)
        encryption_context: CKKS контекст (bytes)
        image_path: Път до файла със снимката
    
    Returns:
        (vehicle_uuid, vehicle_id) - UUID и ID на създаденото превозно средство
    """
    db = get_db()
    
    try:
        # Определяме дали е криптирано
        is_encrypted = encrypted_embedding is not None
        
        # Стъпка 1: Вмъкване на метаданните в таблица vehicles
        vehicle = Vehicle(
            vehicle_uuid=str(uuid.uuid4()),  # Генерираме уникален UUID
            license_plate=license_plate,
            color=color,
            body_type=body_type,
            image_path=image_path,
            is_encrypted=is_encrypted
        )
        db.add(vehicle)
        db.flush()  # Получаваме vehicle.id без да commit-ваме още
        
        # Стъпка 2: Вмъкване на векторния embedding в таблица vehicle_vectors
        vector = VehicleVector(
            vehicle_id=vehicle.id,
            # Ако embedding е numpy array, конвертираме го в list
            embedding=embedding.tolist() if isinstance(embedding, np.ndarray) and not is_encrypted else (embedding if not is_encrypted else None),
            encrypted_embedding=encrypted_embedding,
            encryption_context=encryption_context
        )
        db.add(vector)
        
        # Commit на всичко наведнъж
        db.commit()
        
        print(f"✓ Превозно средство {vehicle.vehicle_uuid} вмъкнато успешно")
        return vehicle.vehicle_uuid, vehicle.id
        
    except Exception as e:
        db.rollback()  # Връщаме промените обратно при грешка
        print(f"✗ Грешка при вмъкване: {e}")
        raise e
    finally:
        db.close()


# ============================================
# SEARCH ОПЕРАЦИИ (Търсене)
# ============================================

def search_similar(
    query_embedding: np.ndarray,
    top_k: int = 10,
    filter_color: Optional[str] = None,
    filter_body_type: Optional[str] = None,
    only_encrypted: bool = False
) -> List[Dict]:
    """
    Търси подобни превозни средства използвайки cosine similarity
    
    Args:
        query_embedding: Query embedding (256 числа)
        top_k: Колко резултата да върне (default: 10)
        filter_color: Филтър по цвят (optional)
        filter_body_type: Филтър по тип каросерия (optional)
        only_encrypted: Дали да търси само криптирани (default: False)
    
    Returns:
        List of dicts: [{"vehicle": {...}, "similarity_score": 0.95}, ...]
    """
    db = get_db()
    
    try:
        # Конвертираме numpy array в list ако е нужно
        query_list = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
        
        # Построяваме SQL query
        # JOIN свързва двете таблици
        # cosine_distance изчислява разстоянието между вектори
        query = db.query(
            Vehicle,
            VehicleVector.embedding.cosine_distance(query_list).label("distance")
        ).join(VehicleVector, Vehicle.id == VehicleVector.vehicle_id)
        
        # Филтрираме само некриптирани записи (не можем да търсим в криптирани без специална логика)
        if not only_encrypted:
            query = query.filter(Vehicle.is_encrypted == False)
            query = query.filter(VehicleVector.embedding.isnot(None))
        
        # Прилагаме допълнителни филтри ако има
        if filter_color:
            query = query.filter(Vehicle.color == filter_color)
        if filter_body_type:
            query = query.filter(Vehicle.body_type == filter_body_type)
        
        # Сортираме по разстояние (най-малкото разстояние = най-подобен)
        # Ограничаваме до top_k резултата
        results = query.order_by(text("distance")).limit(top_k).all()
        
        # Форматираме резултатите
        formatted_results = []
        for vehicle, distance in results:
            formatted_results.append({
                "vehicle": {
                    "id": vehicle.id,
                    "uuid": vehicle.vehicle_uuid,
                    "license_plate": vehicle.license_plate,
                    "color": vehicle.color,
                    "body_type": vehicle.body_type,
                    "image_path": vehicle.image_path,
                    "is_encrypted": vehicle.is_encrypted,
                    "created_at": vehicle.created_at
                },
                "similarity_score": float(1 - distance)  # Конвертираме distance в similarity (0-1)
            })
        
        print(f"✓ Намерени {len(formatted_results)} подобни превозни средства")
        return formatted_results
        
    except Exception as e:
        print(f"✗ Грешка при търсене: {e}")
        raise e
    finally:
        db.close()


# ============================================
# READ ОПЕРАЦИИ (Четене)
# ============================================

def get_all_vehicles(skip: int = 0, limit: int = 100) -> List[Vehicle]:
    """
    Взема всички превозни средства (с pagination)
    
    Args:
        skip: Колко записа да прескочи (за pagination)
        limit: Максимален брой записи да върне
    
    Returns:
        List of Vehicle objects
    """
    db = get_db()
    
    try:
        vehicles = db.query(Vehicle).offset(skip).limit(limit).all()
        print(f"✓ Извлечени {len(vehicles)} превозни средства")
        return vehicles
    except Exception as e:
        print(f"✗ Грешка при извличане: {e}")
        raise e
    finally:
        db.close()


def get_vehicle_by_uuid(vehicle_uuid: str) -> Optional[Vehicle]:
    """
    Взема превозно средство по UUID
    
    Args:
        vehicle_uuid: UUID на превозното средство
    
    Returns:
        Vehicle object или None ако не е намерено
    """
    db = get_db()
    
    try:
        vehicle = db.query(Vehicle).filter(Vehicle.vehicle_uuid == vehicle_uuid).first()
        if vehicle:
            print(f"✓ Превозно средство {vehicle_uuid} намерено")
        else:
            print(f"✗ Превозно средство {vehicle_uuid} не е намерено")
        return vehicle
    except Exception as e:
        print(f"✗ Грешка при търсене по UUID: {e}")
        raise e
    finally:
        db.close()


def get_vehicle_vector(vehicle_id: int) -> Optional[VehicleVector]:
    """
    Взема векторния embedding за дадено превозно средство
    
    Args:
        vehicle_id: ID на превозното средство
    
    Returns:
        VehicleVector object или None
    """
    db = get_db()
    
    try:
        vector = db.query(VehicleVector).filter(VehicleVector.vehicle_id == vehicle_id).first()
        return vector
    except Exception as e:
        print(f"✗ Грешка при извличане на вектор: {e}")
        raise e
    finally:
        db.close()


# ============================================
# DELETE ОПЕРАЦИИ (Изтриване)
# ============================================

def delete_vehicle(vehicle_uuid: str) -> bool:
    """
    Изтрива превозно средство (и неговия вектор автоматично заради CASCADE)
    
    Args:
        vehicle_uuid: UUID на превозното средство
    
    Returns:
        True ако е изтрито успешно, False ако не е намерено
    """
    db = get_db()
    
    try:
        vehicle = db.query(Vehicle).filter(Vehicle.vehicle_uuid == vehicle_uuid).first()
        
        if vehicle:
            db.delete(vehicle)
            db.commit()
            print(f"✓ Превозно средство {vehicle_uuid} изтрито")
            return True
        else:
            print(f"✗ Превозно средство {vehicle_uuid} не е намерено")
            return False
            
    except Exception as e:
        db.rollback()
        print(f"✗ Грешка при изтриване: {e}")
        raise e
    finally:
        db.close()


# ============================================
# STATISTICS ОПЕРАЦИИ (Статистика)
# ============================================

def get_database_stats() -> Dict:
    """
    Взема статистика за базата данни
    
    Returns:
        Dict с различни статистики
    """
    db = get_db()
    
    try:
        # Брой превозни средства
        total_vehicles = db.query(Vehicle).count()
        
        # Брой вектори
        total_vectors = db.query(VehicleVector).count()
        
        # Брой криптирани
        encrypted_count = db.query(Vehicle).filter(Vehicle.is_encrypted == True).count()
        
        # Разпределение по цветове
        colors = db.query(
            Vehicle.color, 
            text("COUNT(*) as count")
        ).group_by(Vehicle.color).all()
        
        # Разпределение по тип каросерия
        body_types = db.query(
            Vehicle.body_type,
            text("COUNT(*) as count")
        ).group_by(Vehicle.body_type).all()
        
        stats = {
            "total_vehicles": total_vehicles,
            "total_vectors": total_vectors,
            "encrypted_count": encrypted_count,
            "plain_count": total_vehicles - encrypted_count,
            "color_distribution": {color: count for color, count in colors if color},
            "body_type_distribution": {body_type: count for body_type, count in body_types if body_type}
        }
        
        return stats
        
    except Exception as e:
        print(f"✗ Грешка при извличане на статистика: {e}")
        raise e
    finally:
        db.close()