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

# ... existing imports ...

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
            license_plate=None,  # Hidden
            color=None,          # Hidden
            body_type=None,      # Hidden
            image_path=None,     # Hidden
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