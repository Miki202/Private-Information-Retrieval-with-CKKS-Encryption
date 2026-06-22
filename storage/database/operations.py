"""
Database operations with TRUE PIR support
"""
from typing import List, Optional, Tuple, Dict
import uuid
import numpy as np
import time

from .models import Vehicle
from .connection import get_db
from .vector_store import get_plain_store, save_all_stores
from .encryption import (
    encrypt_embedding_simple,
    decrypt_embedding_simple,
    normalize_embedding,
    encrypt_metadata,
    decrypt_metadata,
    PIRClient,
    PIRServer,
    homomorphic_dot_product
)

# ============================================
# INSERT OPERATIONS
# ============================================

def insert_vehicle_plain(
    license_plate: Optional[str] = None,
    color: Optional[str] = None,
    body_type: Optional[str] = None,
    embedding: np.ndarray = None,
    image_path: Optional[str] = None,
) -> Tuple[str, int]:
    """Insert vehicle in PLAIN mode (uses FAISS)"""
    
    db = get_db()
    
    try:
        vehicle = Vehicle(
            vehicle_uuid=str(uuid.uuid4()),
            license_plate=license_plate,
            color=color,
            body_type=body_type,
            image_path=image_path,
            is_encrypted=False
        )
        
        db.add(vehicle)
        db.flush()
        
        store = get_plain_store()
        faiss_id = store.add_vector(vehicle.id, embedding)
        vehicle.faiss_id = faiss_id
        
        db.commit()
        save_all_stores()
        
        return vehicle.vehicle_uuid, vehicle.id
        
    except Exception as e:
        db.rollback()
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
    """Insert vehicle in TRUE PIR mode (everything encrypted)"""
    
    db = get_db()
    
    try:
        # Encrypt embedding
        normalized = normalize_embedding(embedding)
        encrypted_data, context_data = encrypt_embedding_simple(normalized)
        
        # Encrypt metadata
        metadata = {
            "license_plate": license_plate,
            "color": color,
            "body_type": body_type,
            "image_path": image_path
        }
        encrypted_meta = encrypt_metadata(metadata, context_data)
        
        # Store with NULL plain metadata
        vehicle = Vehicle(
            vehicle_uuid=str(uuid.uuid4()),
            is_encrypted=True,
            faiss_id=None,
            license_plate=None,
            color=None,
            body_type=None,
            image_path=None,
            encrypted_embedding=encrypted_data,
            encrypted_metadata=encrypted_meta,
            encryption_context=context_data
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


def insert_vehicle(
    license_plate: Optional[str] = None,
    color: Optional[str] = None,
    body_type: Optional[str] = None,
    embedding: Optional[np.ndarray] = None,
    encrypted_embedding: Optional[bytes] = None,
    encryption_context: Optional[bytes] = None,
    image_path: Optional[str] = None,
) -> Tuple[str, int]:
    """Insert vehicle (auto-detects mode based on parameters)"""
    
    if encrypted_embedding is not None:
        db = get_db()
        try:
            vehicle = Vehicle(
                vehicle_uuid=str(uuid.uuid4()),
                license_plate=license_plate,
                color=color,
                body_type=body_type,
                image_path=image_path,
                is_encrypted=True,
                encrypted_embedding=encrypted_embedding,
                encryption_context=encryption_context
            )
            db.add(vehicle)
            db.commit()
            db.refresh(vehicle)
            return vehicle.vehicle_uuid, vehicle.id
        finally:
            db.close()
    else:
        return insert_vehicle_plain(license_plate, color, body_type, embedding, image_path)


# ============================================
# SEARCH OPERATIONS
# ============================================

def search_similar_plain(
    query_embedding: np.ndarray,
    top_k: int = 10,
    filter_color: Optional[str] = None,
    filter_body_type: Optional[str] = None,
) -> List[Dict]:
    """Plain search using FAISS (fast but no privacy)"""
    
    db = get_db()
    
    try:
        store = get_plain_store()
        distances, vehicle_ids = store.search(query_embedding, k=top_k * 3)
        
        results = []
        for dist, vid in zip(distances, vehicle_ids):
            if vid == -1:
                continue
            
            vehicle = db.query(Vehicle).filter(
                Vehicle.id == vid,
                Vehicle.is_encrypted == False
            ).first()
            
            if not vehicle:
                continue
            
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
        
        return results
        
    finally:
        db.close()


def pir_search_true(
    query_embedding: np.ndarray,
    top_k: int = 10,
    filter_color: Optional[str] = None,
    filter_body_type: Optional[str] = None,
    verbose: bool = False
) -> List[Dict]:
    """
    TRUE PIR Search - Fully Homomorphic
    
    Note: Filtering not supported with encrypted metadata
          All encrypted vehicles are scanned
    """
    
    if filter_color or filter_body_type:
        if verbose:
            print("Warning: Filtering not supported with encrypted metadata")
            print("         Returning top-K from all encrypted vehicles")
    
    db = get_db()
    
    try:
        # CLIENT: Prepare encrypted query
        client = PIRClient()
        start_prep = time.time()
        query_data = client.prepare_query(query_embedding)
        prep_time = time.time() - start_prep
        
        encrypted_query = query_data["encrypted_query"]
        query_context = query_data["context"]
        
        if verbose:
            print(f"Client query preparation: {prep_time*1000:.2f}ms")
        
        # SERVER: Get ALL encrypted vehicles (no filtering possible)
        encrypted_vehicles = db.query(Vehicle).filter(
            Vehicle.is_encrypted == True
        ).all()
        
        if len(encrypted_vehicles) == 0:
            return []
        
        if verbose:
            print(f"Server scanning: {len(encrypted_vehicles)} encrypted vehicles")
        
        # SERVER: Homomorphic similarity computation
        encrypted_results = []
        start_server = time.time()
        
        for vehicle in encrypted_vehicles:
            encrypted_sim = homomorphic_dot_product(
                encrypted_query,
                vehicle.encrypted_embedding,
                query_context
            )
            
            encrypted_results.append({
                "vehicle_id": vehicle.id,
                "encrypted_similarity": encrypted_sim,
                "encrypted_metadata": vehicle.encrypted_metadata,
                "context": vehicle.encryption_context
            })
        
        server_time = time.time() - start_server
        
        if verbose:
            print(f"Server homomorphic computation: {server_time*1000:.2f}ms")
        
        # CLIENT: Decrypt similarities
        start_decrypt = time.time()
        
        decrypted_scores = []
        for result in encrypted_results:
            # Decrypt similarity score
            decrypted_sim = decrypt_embedding_simple(
                result["encrypted_similarity"],
                query_context
            )
            score = float(decrypted_sim[0])
            
            decrypted_scores.append({
                "vehicle_id": result["vehicle_id"],
                "similarity_score": score,
                "encrypted_metadata": result["encrypted_metadata"],
                "context": result["context"]
            })
        
        # Sort and select top-k
        decrypted_scores.sort(key=lambda x: x["similarity_score"], reverse=True)
        top_results = decrypted_scores[:top_k]
        
        # CLIENT: Decrypt metadata for top-k only
        final_results = []
        for result in top_results:
            # Decrypt metadata
            metadata = decrypt_metadata(
                result["encrypted_metadata"],
                result["context"]
            )
            
            # Get vehicle record
            vehicle = db.query(Vehicle).filter(
                Vehicle.id == result["vehicle_id"]
            ).first()
            
            if vehicle:
                final_results.append({
                    "vehicle": {
                        "id": vehicle.id,
                        "uuid": vehicle.vehicle_uuid,
                        "license_plate": metadata.get("license_plate"),
                        "color": metadata.get("color"),
                        "body_type": metadata.get("body_type"),
                        "image_path": metadata.get("image_path"),
                        "is_encrypted": True,
                        "created_at": vehicle.created_at,
                        "updated_at": vehicle.updated_at
                    },
                    "similarity_score": result["similarity_score"]
                })
        
        decrypt_time = time.time() - start_decrypt
        
        if verbose:
            total_time = prep_time + server_time + decrypt_time
            print(f"Client decryption: {decrypt_time*1000:.2f}ms")
            print(f"Total PIR search: {total_time*1000:.2f}ms")
        
        return final_results
        
    except Exception as e:
        if verbose:
            print(f"PIR search error: {e}")
        import traceback
        traceback.print_exc()
        return []
    finally:
        db.close()


# Aliases
search_similar = search_similar_plain
pir_search = pir_search_true


# ============================================
# READ OPERATIONS
# ============================================

def get_all_vehicles(
    skip: int = 0,
    limit: int = 100,
    encrypted_only: bool = False,
    plain_only: bool = False
) -> List[Vehicle]:
    """Get vehicles with optional filtering"""
    db = get_db()
    try:
        query = db.query(Vehicle)
        
        if encrypted_only:
            query = query.filter(Vehicle.is_encrypted == True)
        elif plain_only:
            query = query.filter(Vehicle.is_encrypted == False)
        
        return query.offset(skip).limit(limit).all()
    finally:
        db.close()


def get_vehicle_by_uuid(vehicle_uuid: str) -> Optional[Vehicle]:
    """Get vehicle by UUID"""
    db = get_db()
    try:
        return db.query(Vehicle).filter(
            Vehicle.vehicle_uuid == vehicle_uuid
        ).first()
    finally:
        db.close()


# ============================================
# DELETE OPERATIONS
# ============================================

def delete_vehicle(vehicle_uuid: str) -> bool:
    """Delete vehicle by UUID"""
    db = get_db()
    try:
        vehicle = db.query(Vehicle).filter(
            Vehicle.vehicle_uuid == vehicle_uuid
        ).first()
        
        if not vehicle:
            return False
        
        if not vehicle.is_encrypted and vehicle.faiss_id is not None:
            store = get_plain_store()
            store.remove_vector(vehicle.faiss_id)
            save_all_stores()
        
        db.delete(vehicle)
        db.commit()
        
        return True
        
    except Exception as e:
        db.rollback()
        return False
    finally:
        db.close()


# ============================================
# STATISTICS
# ============================================

def get_database_stats() -> Dict:
    """Get database statistics"""
    db = get_db()
    
    try:
        from sqlalchemy import func
        
        total = db.query(Vehicle).count()
        encrypted = db.query(Vehicle).filter(Vehicle.is_encrypted == True).count()
        plain = total - encrypted
        
        colors = db.query(
            Vehicle.color,
            func.count(Vehicle.id)
        ).group_by(Vehicle.color).all()
        
        body_types = db.query(
            Vehicle.body_type,
            func.count(Vehicle.id)
        ).group_by(Vehicle.body_type).all()
        
        encrypted_vehicles = db.query(Vehicle).filter(
            Vehicle.is_encrypted == True
        ).all()
        
        total_encrypted_size = sum(v.get_storage_size() for v in encrypted_vehicles)
        avg_encrypted_size = total_encrypted_size / max(encrypted, 1)
        
        plain_store = get_plain_store()
        
        return {
            "total_vehicles": total,
            "plain_count": plain,
            "encrypted_count": encrypted,
            "color_distribution": {c: cnt for c, cnt in colors if c},
            "body_type_distribution": {b: cnt for b, cnt in body_types if b},
            "faiss_plain_vectors": plain_store.get_stats()["total_vectors"],
            "total_encrypted_storage_bytes": total_encrypted_size,
            "avg_encrypted_storage_bytes": int(avg_encrypted_size),
        }
        
    finally:
        db.close()


def get_encryption_stats() -> Dict:
    """Get encryption statistics"""
    stats = get_database_stats()
    plain_size = 256 * 4
    
    return {
        "total_vehicles": stats["total_vehicles"],
        "encrypted_count": stats["encrypted_count"],
        "plain_count": stats["plain_count"],
        "encryption_percentage": (stats["encrypted_count"] / max(stats["total_vehicles"], 1)) * 100,
        "avg_encrypted_size_bytes": stats["avg_encrypted_size_bytes"],
        "plain_embedding_size_bytes": plain_size,
        "storage_overhead": stats["avg_encrypted_storage_bytes"] / plain_size if plain_size else 0
    }