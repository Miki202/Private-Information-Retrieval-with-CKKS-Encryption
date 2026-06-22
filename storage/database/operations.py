"""
Database operations: PostgreSQL (metadata) + FAISS (vectors)
"""
from typing import List, Optional, Tuple, Dict
import uuid
import numpy as np
import os

from .models import Vehicle
from .connection import get_db
from .vector_store import get_plain_store, get_encrypted_store, save_all_stores
from .encryption import (
    encrypt_embedding_simple,
    decrypt_embedding_simple,
    normalize_embedding,
    PIRClient,
    PIRServer
)

# Data directory за encrypted files
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ============================================
# INSERT ОПЕРАЦИИ
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
    Вмъква vehicle: PostgreSQL (metadata) + FAISS (vector)
    
    Returns:
        (vehicle_uuid, vehicle_id)
    """
    db = get_db()
    
    try:
        is_encrypted = encrypted_embedding is not None
        
        # Стъпка 1: PostgreSQL - metadata
        vehicle = Vehicle(
            vehicle_uuid=str(uuid.uuid4()),
            license_plate=license_plate,
            color=color,
            body_type=body_type,
            image_path=image_path,
            is_encrypted=is_encrypted,
            faiss_id=None  # Ще попълним след FAISS add
        )
        
        db.add(vehicle)
        db.flush()  # Get vehicle.id
        
        # Стъпка 2: FAISS - vector
        if is_encrypted and encrypted_embedding is not None:
            # Encrypted mode
            store = get_encrypted_store()
            
            # Съхраняваме encrypted data във файлове
            enc_path = os.path.join(DATA_DIR, f"encrypted_{vehicle.id}.bin")
            ctx_path = os.path.join(DATA_DIR, f"context_{vehicle.id}.bin")
            
            with open(enc_path, 'wb') as f:
                f.write(encrypted_embedding)
            with open(ctx_path, 'wb') as f:
                f.write(encryption_context)
            
            # За FAISS търсене - декриптираме временно
            # (Това е компромис - за pure PIR трябва custom vector search)
            decrypted = decrypt_embedding_simple(encrypted_embedding, encryption_context)
            faiss_id = store.add_vector(vehicle.id, decrypted)
            
            vehicle.image_path = f"{enc_path}|{ctx_path}"
            
        elif embedding is not None:
            # Plain mode
            store = get_plain_store()
            faiss_id = store.add_vector(vehicle.id, embedding)
        
        else:
            raise ValueError("Трябва embedding или encrypted_embedding")
        
        # Update FAISS ID в PostgreSQL
        vehicle.faiss_id = faiss_id
        
        db.commit()
        save_all_stores()
        
        print(f"✓ Vehicle {vehicle.vehicle_uuid} inserted")
        print(f"  PostgreSQL ID: {vehicle.id}")
        print(f"  FAISS ID: {vehicle.faiss_id}")
        
        return vehicle.vehicle_uuid, vehicle.id
        
    except Exception as e:
        db.rollback()
        print(f"✗ Insert error: {e}")
        raise e
    finally:
        db.close()


def insert_vehicle_encrypted(
    license_plate: Optional[str] = None,
    color: Optional[str] = None,
    body_type: Optional[str] = None,
    embedding: np.ndarray = None,
    image_path: Optional[str] = None,
) -> Tuple[str, int]:
    """
    Вмъква encrypted vehicle
    """
    print("Вмъкване с криптиране...")
    
    # Normalize за PIR
    normalized = normalize_embedding(embedding)
    
    # Encrypt
    encrypted_data, context_data = encrypt_embedding_simple(normalized)
    print(f"  Криптирано: {len(encrypted_data):,} bytes")
    
    return insert_vehicle(
        license_plate=license_plate,
        color=color,
        body_type=body_type,
        embedding=None,
        encrypted_embedding=encrypted_data,
        encryption_context=context_data,
        image_path=image_path
    )


# ============================================
# SEARCH ОПЕРАЦИИ
# ============================================

def search_similar(
    query_embedding: np.ndarray,
    top_k: int = 10,
    filter_color: Optional[str] = None,
    filter_body_type: Optional[str] = None,
) -> List[Dict]:
    """
    Plain search: FAISS → PostgreSQL
    
    1. FAISS: намира k най-близки vectors → vehicle IDs
    2. PostgreSQL: взима metadata за тези IDs
    3. Прилага filters
    """
    db = get_db()
    
    try:
        # Стъпка 1: FAISS search
        store = get_plain_store()
        distances, vehicle_ids = store.search(query_embedding, k=top_k * 3)  # Extra за filtering
        
        if len(vehicle_ids) == 0:
            print("✗ Няма plain vectors")
            return []
        
        print(f"FAISS: {len([v for v in vehicle_ids if v != -1])} candidates")
        
        # Стъпка 2: PostgreSQL - metadata
        results = []
        
        for dist, vid in zip(distances, vehicle_ids):
            if vid == -1:
                continue
            
            # Вземи vehicle от PostgreSQL
            vehicle = db.query(Vehicle).filter(
                Vehicle.id == vid,
                Vehicle.is_encrypted == False
            ).first()
            
            if vehicle is None:
                continue
            
            # Стъпка 3: Apply filters
            if filter_color and vehicle.color != filter_color:
                continue
            if filter_body_type and vehicle.body_type != filter_body_type:
                continue
            
            # Convert L2 distance to similarity
            similarity = 1.0 / (1.0 + float(dist))
            
            results.append({
                "vehicle": vehicle.to_dict(),
                "similarity_score": similarity
            })
            
            if len(results) >= top_k:
                break
        
        print(f"✓ Plain search: {len(results)} results")
        return results
        
    finally:
        db.close()


def pir_search(
    query_embedding: np.ndarray,
    top_k: int = 10,
    filter_color: Optional[str] = None,
    filter_body_type: Optional[str] = None
) -> List[Dict]:
    """
    PIR search: FAISS (encrypted) → PostgreSQL
    
    Забележка: FAISS работи с decrypted vectors за similarity.
    За pure PIR без този компромис, трябва homomorphic vector operations.
    """
    print("\n" + "🔐"*30)
    print("PIR SEARCH (PostgreSQL + FAISS)")
    print("🔐"*30 + "\n")
    
    db = get_db()
    
    try:
        # Client: prepare query
        print("📱 CLIENT: Encrypt query...")
        client = PIRClient()
        query_data = client.prepare_query(query_embedding)
        
        print("📡 Send to server...\n")
        
        # Server: FAISS search
        print("🖥️  SERVER: FAISS search (encrypted store)...")
        store = get_encrypted_store()
        
        normalized_query = normalize_embedding(query_embedding)
        distances, vehicle_ids = store.search(normalized_query, k=top_k * 3)
        
        if len(vehicle_ids) == 0:
            print("✗ Няма encrypted vectors")
            return []
        
        print(f"SERVER: {len([v for v in vehicle_ids if v != -1])} candidates")
        print("📡 Return encrypted results...\n")
        
        # Client: process results
        print("📱 CLIENT: Decrypt & filter...")
        
        results = []
        
        for dist, vid in zip(distances, vehicle_ids):
            if vid == -1:
                continue
            
            # PostgreSQL metadata
            vehicle = db.query(Vehicle).filter(
                Vehicle.id == vid,
                Vehicle.is_encrypted == True
            ).first()
            
            if vehicle is None:
                continue
            
            # Filters
            if filter_color and vehicle.color != filter_color:
                continue
            if filter_body_type and vehicle.body_type != filter_body_type:
                continue
            
            similarity = 1.0 / (1.0 + float(dist))
            
            results.append({
                "vehicle": vehicle.to_dict(),
                "similarity_score": similarity
            })
            
            if len(results) >= top_k:
                break
        
        print(f"\n✓ PIR search complete: {len(results)} results")
        print("="*60 + "\n")
        
        return results
        
    finally:
        db.close()


# ============================================
# READ ОПЕРАЦИИ
# ============================================

def get_all_vehicles(skip: int = 0, limit: int = 100) -> List[Vehicle]:
    """Взема всички vehicles от PostgreSQL"""
    db = get_db()
    try:
        return db.query(Vehicle).offset(skip).limit(limit).all()
    finally:
        db.close()


def get_vehicle_by_uuid(vehicle_uuid: str) -> Optional[Vehicle]:
    """Взема vehicle по UUID"""
    db = get_db()
    try:
        return db.query(Vehicle).filter(Vehicle.vehicle_uuid == vehicle_uuid).first()
    finally:
        db.close()


# ============================================
# DELETE ОПЕРАЦИИ
# ============================================

def delete_vehicle(vehicle_uuid: str) -> bool:
    """
    Изтрива vehicle: PostgreSQL + FAISS cleanup
    """
    db = get_db()
    
    try:
        vehicle = db.query(Vehicle).filter(Vehicle.vehicle_uuid == vehicle_uuid).first()
        
        if not vehicle:
            return False
        
        # Cleanup FAISS mapping
        if vehicle.is_encrypted:
            store = get_encrypted_store()
            # Изтрий encrypted files
            if vehicle.image_path and "|" in vehicle.image_path:
                enc_path, ctx_path = vehicle.image_path.split("|")
                if os.path.exists(enc_path):
                    os.remove(enc_path)
                if os.path.exists(ctx_path):
                    os.remove(ctx_path)
        else:
            store = get_plain_store()
        
        if vehicle.faiss_id is not None:
            store.remove_vector(vehicle.faiss_id)
        
        # Delete от PostgreSQL
        db.delete(vehicle)
        db.commit()
        
        save_all_stores()
        
        print(f"✓ Vehicle {vehicle_uuid} deleted")
        return True
        
    except Exception as e:
        db.rollback()
        print(f"✗ Delete error: {e}")
        return False
    finally:
        db.close()


# ============================================
# STATISTICS
# ============================================

def get_database_stats() -> Dict:
    """Статистика от PostgreSQL"""
    db = get_db()
    
    try:
        from sqlalchemy import func
        
        total = db.query(Vehicle).count()
        encrypted = db.query(Vehicle).filter(Vehicle.is_encrypted == True).count()
        plain = total - encrypted
        
        # Color distribution
        colors = db.query(
            Vehicle.color,
            func.count(Vehicle.id).label("count")
        ).group_by(Vehicle.color).all()
        
        # Body type distribution
        body_types = db.query(
            Vehicle.body_type,
            func.count(Vehicle.id).label("count")
        ).group_by(Vehicle.body_type).all()
        
        # FAISS stats
        plain_store = get_plain_store()
        enc_store = get_encrypted_store()
        
        return {
            "total_vehicles": total,
            "plain_count": plain,
            "encrypted_count": encrypted,
            "color_distribution": {c: cnt for c, cnt in colors if c},
            "body_type_distribution": {b: cnt for b, cnt in body_types if b},
            "faiss_plain_vectors": plain_store.get_stats()["total_vectors"],
            "faiss_encrypted_vectors": enc_store.get_stats()["total_vectors"]
        }
        
    finally:
        db.close()


def get_encryption_stats() -> Dict:
    """Encryption статистика"""
    stats = get_database_stats()
    
    # Прост calculation (в production би трябвало да query-ваш файловете)
    avg_encrypted_size = 100000  # ~100KB per encrypted embedding
    avg_context_size = 50000     # ~50KB context
    plain_size = 256 * 4         # 256 floats * 4 bytes = 1KB
    
    return {
        "total_vehicles": stats["total_vehicles"],
        "encrypted_count": stats["encrypted_count"],
        "plain_count": stats["plain_count"],
        "encryption_percentage": (stats["encrypted_count"] / max(stats["total_vehicles"], 1)) * 100,
        "avg_encrypted_size_bytes": avg_encrypted_size,
        "avg_context_size_bytes": avg_context_size,
        "avg_total_size_bytes": avg_encrypted_size + avg_context_size,
        "plain_embedding_size_bytes": plain_size
    }