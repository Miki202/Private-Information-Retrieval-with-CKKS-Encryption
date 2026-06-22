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

# ============================================
# ENCRYPTED OPERATIONS (Криптирани операции)
# ============================================

from .encryption import encrypt_embedding_simple, decrypt_embedding_simple, get_encryptor

def insert_vehicle_encrypted(
    license_plate: Optional[str] = None,
    color: Optional[str] = None,
    body_type: Optional[str] = None,
    embedding: np.ndarray = None,
    image_path: Optional[str] = None,
) -> Tuple[str, int]:
    """
    Вмъква превозно средство с КРИПТИРАН embedding
    Embedding се normalize-ва автоматично за PIR compatibility
    """
    from .encryption import normalize_embedding, encrypt_embedding_simple
    
    print(f"Вмъкване на превозно средство с криптиране (PIR-ready)...")
    
    # ВАЖНО: Normalize embedding за dot product similarity
    normalized_embedding = normalize_embedding(embedding)
    print(f"  Embedding normalized: ||v|| = {np.linalg.norm(normalized_embedding):.6f}")
    
    # Криптираме normalized embedding
    encrypted_data, context_data = encrypt_embedding_simple(normalized_embedding)
    print(f"  Embedding криптиран: {len(encrypted_data):,} bytes")
    
    # Вмъкваме
    uuid, vid = insert_vehicle(
        license_plate=license_plate,
        color=color,
        body_type=body_type,
        embedding=None,
        encrypted_embedding=encrypted_data,
        encryption_context=context_data,
        image_path=image_path
    )
    
    print(f"✓ Превозно средство {uuid} вмъкнато (PIR-ready)")
    return uuid, vid


def get_decrypted_embedding(vehicle_id: int) -> Optional[np.ndarray]:
    """
    Взема и декриптира embedding за дадено превозно средство
    
    Args:
        vehicle_id: ID на превозното средство
    
    Returns:
        Декриптиран embedding или None ако не съществува
    """
    db = get_db()
    
    try:
        # Вземаме векторните данни
        vector = db.query(VehicleVector).filter(VehicleVector.vehicle_id == vehicle_id).first()
        
        if not vector:
            print(f"✗ Вектор за vehicle_id={vehicle_id} не е намерен")
            return None
        
        # Ако има plain embedding, връщаме го директно
        if vector.embedding is not None:
            print(f"✓ Връщане на plain embedding за vehicle_id={vehicle_id}")
            return np.array(vector.embedding)
        
        # Ако има криптиран embedding, декриптираме го
        if vector.encrypted_embedding is not None and vector.encryption_context is not None:
            print(f"Декриптиране на embedding за vehicle_id={vehicle_id}...")
            embedding = decrypt_embedding_simple(
                vector.encrypted_embedding,
                vector.encryption_context
            )
            print(f"✓ Embedding декриптиран")
            return embedding
        
        print(f"✗ Няма нито plain нито криптиран embedding за vehicle_id={vehicle_id}")
        return None
        
    except Exception as e:
        print(f"✗ Грешка при извличане/декриптиране: {e}")
        raise e
    finally:
        db.close()


def search_similar_encrypted(
    query_embedding: np.ndarray,
    top_k: int = 10,
    filter_color: Optional[str] = None,
    filter_body_type: Optional[str] = None,
) -> List[Dict]:
    """
    Търси подобни превозни средства сред КРИПТИРАНИТЕ записи
    
    ВАЖНО: Това е "naive" имплементация - декриптираме всички записи
           и после правим similarity търсене. Не е истински PIR!
    
    За истински PIR трябва homomorphic операции върху криптираните данни,
    което е много по-сложно и бавно.
    
    Args:
        query_embedding: Query embedding (некриптиран)
        top_k: Брой резултати
        filter_color: Филтър по цвят
        filter_body_type: Филтър по тип
    
    Returns:
        List of dicts с резултати
    """
    print(f"\nТърсене сред криптирани записи...")
    print(f"⚠️  ВНИМАНИЕ: Това е naive имплементация (decrypt-then-search)")
    print(f"   За истински PIR трябва homomorphic comparison операции\n")
    
    db = get_db()
    
    try:
        # Стъпка 1: Вземаме всички криптирани превозни средства
        query = db.query(Vehicle, VehicleVector).join(
            VehicleVector, Vehicle.id == VehicleVector.vehicle_id
        ).filter(Vehicle.is_encrypted == True)
        
        # Прилагаме филтри
        if filter_color:
            query = query.filter(Vehicle.color == filter_color)
        if filter_body_type:
            query = query.filter(Vehicle.body_type == filter_body_type)
        
        results = query.all()
        
        if not results:
            print("✗ Няма криптирани записи в базата")
            return []
        
        print(f"Намерени {len(results)} криптирани записа")
        
        # Стъпка 2: Декриптираме всички embeddings и изчисляваме similarity
        similarities = []
        
        for vehicle, vector in results:
            try:
                # Декриптираме
                decrypted_emb = decrypt_embedding_simple(
                    vector.encrypted_embedding,
                    vector.encryption_context
                )
                
                # Изчисляваме cosine similarity
                # similarity = (A · B) / (||A|| × ||B||)
                dot_product = np.dot(query_embedding, decrypted_emb)
                norm_query = np.linalg.norm(query_embedding)
                norm_db = np.linalg.norm(decrypted_emb)
                similarity = dot_product / (norm_query * norm_db)
                
                similarities.append({
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
                    "similarity_score": float(similarity)
                })
                
            except Exception as e:
                print(f"✗ Грешка при декриптиране на vehicle {vehicle.id}: {e}")
                continue
        
        # Стъпка 3: Сортираме по similarity и вземаме top_k
        similarities.sort(key=lambda x: x["similarity_score"], reverse=True)
        top_results = similarities[:top_k]
        
        print(f"✓ Намерени {len(top_results)} най-подобни превозни средства")
        
        return top_results
        
    except Exception as e:
        print(f"✗ Грешка при криптирано търсене: {e}")
        raise e
    finally:
        db.close()


def convert_to_encrypted(vehicle_uuid: str) -> bool:
    """
    Конвертира съществуващо превозно средство от plain в encrypted
    
    Args:
        vehicle_uuid: UUID на превозното средство
    
    Returns:
        True ако е успешно
    """
    db = get_db()
    
    try:
        # Вземаме превозното средство
        vehicle = db.query(Vehicle).filter(Vehicle.vehicle_uuid == vehicle_uuid).first()
        
        if not vehicle:
            print(f"✗ Превозно средство {vehicle_uuid} не е намерено")
            return False
        
        if vehicle.is_encrypted:
            print(f"⚠️  Превозно средство {vehicle_uuid} вече е криптирано")
            return True
        
        # Вземаме plain embedding
        vector = db.query(VehicleVector).filter(VehicleVector.vehicle_id == vehicle.id).first()
        
        if not vector or vector.embedding is None:
            print(f"✗ Няма plain embedding за vehicle {vehicle_uuid}")
            return False
        
        plain_embedding = np.array(vector.embedding)
        
        # Криптираме
        print(f"Криптиране на превозно средство {vehicle_uuid}...")
        encrypted_data, context_data = encrypt_embedding_simple(plain_embedding)
        
        # Обновяваме записа
        vector.encrypted_embedding = encrypted_data
        vector.encryption_context = context_data
        vector.embedding = None  # Изтриваме plain версията
        
        vehicle.is_encrypted = True
        
        db.commit()
        
        print(f"✓ Превозно средство {vehicle_uuid} конвертирано в encrypted")
        return True
        
    except Exception as e:
        db.rollback()
        print(f"✗ Грешка при конвертиране: {e}")
        raise e
    finally:
        db.close()


def get_encryption_stats() -> Dict:
    """
    Връща статистика за криптирането
    
    Returns:
        Dict със статистики
    """
    db = get_db()
    
    try:
        total = db.query(Vehicle).count()
        encrypted = db.query(Vehicle).filter(Vehicle.is_encrypted == True).count()
        plain = total - encrypted
        
        # Средни размери
        encrypted_vectors = db.query(VehicleVector).filter(
            VehicleVector.encrypted_embedding.isnot(None)
        ).all()
        
        if encrypted_vectors:
            avg_encrypted_size = np.mean([len(v.encrypted_embedding) for v in encrypted_vectors])
            avg_context_size = np.mean([len(v.encryption_context) for v in encrypted_vectors])
        else:
            avg_encrypted_size = 0
            avg_context_size = 0
        
        return {
            "total_vehicles": total,
            "encrypted_count": encrypted,
            "plain_count": plain,
            "encryption_percentage": (encrypted / total * 100) if total > 0 else 0,
            "avg_encrypted_size_bytes": int(avg_encrypted_size),
            "avg_context_size_bytes": int(avg_context_size),
            "avg_total_size_bytes": int(avg_encrypted_size + avg_context_size),
            "plain_embedding_size_bytes": 256 * 8  # 256 floats × 8 bytes
        }
        
    finally:
        db.close()

from .encryption import PIRClient, PIRServer

def pir_search(
    query_embedding: np.ndarray,
    top_k: int = 10,
    filter_color: Optional[str] = None,
    filter_body_type: Optional[str] = None
) -> List[Dict]:
    """
    🔐 ИСТИНСКИ PRIVATE INFORMATION RETRIEVAL ТЪРСЕНЕ 🔐
    
    Client криптира query → Server прави homomorphic операции → Client декриптира
    Server НИКОГА не вижда query или similarity scores!
    
    Args:
        query_embedding: Query vector (256-dim numpy array)
        top_k: Брой резултати
        filter_color: Optional color filter
        filter_body_type: Optional body type filter
    
    Returns:
        List of dicts със vehicle данни и similarity scores
    """
    print("\n" + "🔐"*30)
    print("ЗАПОЧВАНЕ НА PIR ТЪРСЕНЕ")
    print("🔐"*30 + "\n")
    
    # ============ CLIENT SIDE ============
    print("📱 CLIENT: Подготовка на query...")
    client = PIRClient()
    query_data = client.prepare_query(query_embedding)
    
    # ============ NETWORK TRANSFER (simulated) ============
    print("\n📡 Изпращане на encrypted query към server...\n")
    
    # ============ SERVER SIDE ============
    print("🖥️  SERVER: Обработка...")
    server = PIRServer()
    encrypted_results, vehicle_ids = server.search_encrypted_database(
        encrypted_query=query_data["encrypted_query"],
        query_context=query_data["context"],
        filter_color=filter_color,
        filter_body_type=filter_body_type
    )
    
    # ============ NETWORK TRANSFER (simulated) ============
    print("📡 Връщане на encrypted results към client...\n")
    
    # ============ CLIENT SIDE ============
    print("📱 CLIENT: Декриптиране и подбор на top-K...")
    top_results = client.process_results(
        encrypted_results=encrypted_results,
        vehicle_ids=vehicle_ids,
        context_bytes=query_data["context"],
        top_k=top_k
    )
    
    # Вземаме пълните метаданни за top-K results
    print(f"\n📱 CLIENT: Заявка за метаданни на top {len(top_results)} results...")
    db = get_db()
    
    try:
        final_results = []
        for result in top_results:
            vehicle = db.query(Vehicle).filter(Vehicle.id == result["vehicle_id"]).first()
            if vehicle:
                final_results.append({
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
                    "similarity_score": result["similarity_score"]
                })
        
        print("\n" + "🔐"*30)
        print(f"✓ PIR ТЪРСЕНЕ ЗАВЪРШЕНО - {len(final_results)} резултата")
        print("✓ Query остана НАПЪЛНО СКРИТ от server!")
        print("🔐"*30 + "\n")
        
        return final_results
        
    finally:
        db.close()