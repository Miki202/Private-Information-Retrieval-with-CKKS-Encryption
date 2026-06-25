"""
Database operations за TRUE PIR с plaintext за UI
"""
from typing import List, Optional, Tuple, Dict
import uuid
import numpy as np
import time

from .models import Vehicle
from .connection import get_db
from .encryption import (
    encrypt_embedding_simple,
    decrypt_embedding_simple,
    normalize_embedding,
    encrypt_metadata,
    decrypt_metadata,
    PIRClient,
    homomorphic_dot_product,
)

def insert_vehicle(
    embedding: np.ndarray,
    license_plate: Optional[str] = None,
    color: Optional[str] = None,
    body_type: Optional[str] = None,
    image_path: Optional[str] = None,
) -> Tuple[str, int]:
    """
    Вмъква превозно средство с криптирани данни + plaintext за UI.
    """
    db = get_db()
    
    try:
        if embedding is None:
            raise ValueError("Ембединга не може да е None")
        
        # Криптиране на embedding
        normalized = normalize_embedding(embedding)
        encrypted_embedding = encrypt_embedding_simple(normalized)
        
        # Криптиране на метаданни
        metadata = {
            "license_plate": license_plate,
            "color": color,
            "body_type": body_type,
            "image_path": image_path,
        }
        encrypted_metadata = encrypt_metadata(metadata)
        
        # Създаване на запис
        vehicle = Vehicle(
            vehicle_uuid=str(uuid.uuid4()),
            encrypted_embedding=encrypted_embedding,
            encrypted_metadata=encrypted_metadata,
<<<<<<< HEAD
            license_plate=license_plate,  
            color=color,                  
            body_type=body_type,          
            image_path=image_path,        
=======
            license_plate=license_plate,  # Plaintext
            color=color,                  # Plaintext
            body_type=body_type,          # Plaintext
            image_path=image_path,        # Plaintext
>>>>>>> 53c138d07e9fe2796f830eb79d6f610bd9146439
        )
        
        db.add(vehicle)
        db.commit()
        db.refresh(vehicle)
        
        return vehicle.vehicle_uuid, vehicle.id
        
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()


def pir_search_true(
    query_embedding: np.ndarray,
    top_k: int = 10,
    verbose: bool = False,
) -> List[Dict]:
<<<<<<< HEAD
=======
    """
    TRUE PIR търсене - използва САМО криптирани данни.
    """
>>>>>>> 53c138d07e9fe2796f830eb79d6f610bd9146439
    db = get_db()
    
    try:
        client = PIRClient()
        
        # Клиент: криптиране на заявката
        start = time.time()
        query_data = client.prepare_query(query_embedding)
        prep_time = time.time() - start
        
        encrypted_query = query_data["encrypted_query"]
        
        # Сървър: взима всички криптирани записи
        vehicles = db.query(Vehicle).all()
        
        if verbose:
<<<<<<< HEAD
            print(f"Подготовка на заявката: {prep_time*1000:.2f}ms")
            print(f"Сканиране от сървъра: {len(vehicles)} vehicles")
=======
            print(f"Query prep: {prep_time*1000:.2f}ms")
            print(f"Server scanning: {len(vehicles)} vehicles")
>>>>>>> 53c138d07e9fe2796f830eb79d6f610bd9146439
        
        if not vehicles:
            return []
        
        # Сървър: хомоморфно изчисление
        start_server = time.time()
        
        encrypted_results = []
        
        for v in vehicles:
            enc_sim = homomorphic_dot_product(
                encrypted_query,
                v.encrypted_embedding
            )
            
            encrypted_results.append({
                "vehicle_id": v.id,
                "encrypted_similarity": enc_sim,
                "encrypted_metadata": v.encrypted_metadata,
            })
        
        server_time = time.time() - start_server
        
        # Клиент: декриптиране на резултати
        decrypted = []
        
        for r in encrypted_results:
            sim = decrypt_embedding_simple(r["encrypted_similarity"])
<<<<<<< HEAD
            score = float(sim[0])
=======
            score = float(sim[1])
>>>>>>> 53c138d07e9fe2796f830eb79d6f610bd9146439
            
            decrypted.append({
                "vehicle_id": r["vehicle_id"],
                "score": score,
                "encrypted_metadata": r["encrypted_metadata"],
            })
        
        # Сортиране и избор на Top-K
        decrypted.sort(key=lambda x: x["score"], reverse=True)
        top_results = decrypted[:top_k]
        
        # Декриптиране на метаданни само за Top-K
        final_results = []
        
        for r in top_results:
            meta = decrypt_metadata(r["encrypted_metadata"])
            
            v = db.query(Vehicle).filter(
                Vehicle.id == r["vehicle_id"]
            ).first()
            
            if v:
                final_results.append({
                    "vehicle": {
                        "id": v.id,
                        "uuid": v.vehicle_uuid,
                        "license_plate": meta.get("license_plate"),
                        "color": meta.get("color"),
                        "body_type": meta.get("body_type"),
                        "image_path": meta.get("image_path"),
                    },
                    "similarity_score": r["score"],
                })
        
        if verbose:
            total = prep_time + server_time
            print(f"Server time: {server_time*1000:.2f}ms")
<<<<<<< HEAD
            print(f"Общо: {total*1000:.2f}ms")
=======
            print(f"Total PIR: {total*1000:.2f}ms")
>>>>>>> 53c138d07e9fe2796f830eb79d6f610bd9146439
        
        return final_results
        
    finally:
        db.close()


pir_search = pir_search_true


def get_all_vehicles(skip: int = 0, limit: int = 100) -> List[Vehicle]:
    """Връща всички записи"""
    db = get_db()
    try:
        return db.query(Vehicle).offset(skip).limit(limit).all()
    finally:
        db.close()


def get_vehicle_by_uuid(vehicle_uuid: str) -> Optional[Vehicle]:
    """Връща запис по UUID"""
    db = get_db()
    try:
        return db.query(Vehicle).filter(
            Vehicle.vehicle_uuid == vehicle_uuid
        ).first()
    finally:
        db.close()


def delete_vehicle(vehicle_uuid: str) -> bool:
    """Изтриване на запис по UUID"""
    db = get_db()
    
    try:
        vehicle = db.query(Vehicle).filter(
            Vehicle.vehicle_uuid == vehicle_uuid
        ).first()
        
        if not vehicle:
            return False
        
        db.delete(vehicle)
        db.commit()
        return True
        
    except Exception:
        db.rollback()
        return False
    finally:
        db.close()


def get_database_stats() -> Dict:
<<<<<<< HEAD
=======
    """Статистика за базата данни"""
>>>>>>> 53c138d07e9fe2796f830eb79d6f610bd9146439
    db = get_db()
    
    try:
        total = db.query(Vehicle).count()
        
        vehicles = db.query(Vehicle).all()
        
        total_size = sum(v.get_storage_size() for v in vehicles)
        avg_size = total_size / max(total, 1)
        
        return {
            "total_vehicles": total,
            "avg_encrypted_storage_bytes": int(avg_size),
        }
        
    finally:
        db.close()


def get_encryption_stats() -> Dict:
<<<<<<< HEAD
=======
    """Статистика за криптирането"""
>>>>>>> 53c138d07e9fe2796f830eb79d6f610bd9146439
    stats = get_database_stats()
    
    plain_size = 256 * 4
    
    return {
        "total_vehicles": stats["total_vehicles"],
        "plain_embedding_size_bytes": plain_size,
        "storage_overhead": stats["avg_encrypted_storage_bytes"] / plain_size if plain_size else 0,
    }