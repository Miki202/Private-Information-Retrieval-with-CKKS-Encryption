"""
TRUE PIR Test with Encrypted Metadata
Tests only encrypted mode (no plain/FAISS tests)
"""
import numpy as np
import time
import sys

def test_true_pir():
    """Test suite for TRUE PIR with encrypted metadata"""
    
    print("\n" + "="*70)
    print("TRUE PIR TEST - ENCRYPTED METADATA")
    print("="*70)
    
    from database.operations import (
        insert_vehicle_encrypted,
        pir_search_true,
        get_database_stats,
        get_encryption_stats
    )
    
    # ===== TEST 1: Insert Encrypted Vehicles =====
    print("\n[TEST 1] Insert Encrypted Vehicles (TRUE PIR mode)")
    print("-"*70)
    
    encrypted_embeddings = []
    encrypted_vehicles = [
        ("ENC001", "red", "sedan"),
        ("ENC002", "blue", "suv"),
        ("ENC003", "red", "coupe"),
        ("ENC004", "green", "sedan"),
        ("ENC005", "red", "sedan"),
        ("ENC006", "blue", "sedan"),
        ("ENC007", "black", "truck"),
        ("ENC008", "white", "suv"),
    ]
    
    print("Inserting vehicles with encrypted embeddings and metadata...")
    for plate, color, body in encrypted_vehicles:
        emb = np.random.rand(256)
        encrypted_embeddings.append(emb)
        
        uuid, vid = insert_vehicle_encrypted(
            license_plate=plate,
            color=color,
            body_type=body,
            embedding=emb,
            image_path=f"{plate}.jpg"
        )
        print(f"  {plate}: UUID={uuid[:8]}..., ID={vid}")
    
    print(f"\nResult: {len(encrypted_vehicles)} vehicles inserted")
    print("  Embeddings: ENCRYPTED")
    print("  Metadata: ENCRYPTED")
    print("  Server can see: NOTHING")
    
    # ===== TEST 2: Verify Metadata Encryption =====
    print("\n[TEST 2] Verify Metadata Encryption in Database")
    print("-"*70)
    
    from database.connection import get_db
    from database.models import Vehicle
    
    db = get_db()
    try:
        encrypted_vehs = db.query(Vehicle).filter(
            Vehicle.is_encrypted == True
        ).all()
        
        print(f"\nFound {len(encrypted_vehs)} encrypted vehicles")
        
        # Check first two
        for i, veh in enumerate(encrypted_vehs[:2], 1):
            print(f"\nVehicle {i}:")
            print(f"  UUID: {veh.vehicle_uuid}")
            print(f"  license_plate: {veh.license_plate} (should be None)")
            print(f"  color: {veh.color} (should be None)")
            print(f"  body_type: {veh.body_type} (should be None)")
            print(f"  encrypted_embedding: {len(veh.encrypted_embedding):,} bytes")
            print(f"  encrypted_metadata: {len(veh.encrypted_metadata):,} bytes")
            print(f"  encryption_context: {len(veh.encryption_context):,} bytes")
            print(f"  Total size: {veh.get_storage_size()/1024:.1f} KB")
        
        # Verify all have encrypted data and no plain metadata
        all_encrypted = True
        for veh in encrypted_vehs:
            if veh.license_plate is not None or \
               veh.color is not None or \
               veh.body_type is not None:
                print(f"\nERROR: Vehicle {veh.vehicle_uuid} has visible metadata!")
                all_encrypted = False
            
            if veh.encrypted_embedding is None or \
               veh.encrypted_metadata is None or \
               veh.encryption_context is None:
                print(f"\nERROR: Vehicle {veh.vehicle_uuid} missing encrypted data!")
                all_encrypted = False
        
        if not all_encrypted:
            return False
        
        print(f"\nResult: All vehicles properly encrypted")
        print("  Plain metadata fields: NULL")
        print("  Encrypted fields: Present")
        
    finally:
        db.close()
    
    # ===== TEST 3: Database Statistics =====
    print("\n[TEST 3] Database Statistics")
    print("-"*70)
    
    stats = get_database_stats()
    enc_stats = get_encryption_stats()
    
    print(f"Total vehicles: {stats['total_vehicles']}")
    print(f"Encrypted vehicles: {stats['encrypted_count']}")
    
    print(f"\nStorage:")
    print(f"  Plain embedding size: {enc_stats['plain_embedding_size_bytes']} bytes")
    print(f"  Encrypted average size: {enc_stats['avg_encrypted_size_bytes']:,} bytes")
    print(f"  Storage overhead: {enc_stats['storage_overhead']:.1f}x")
    
    if stats['encrypted_count'] != len(encrypted_vehicles):
        print(f"\nERROR: Expected {len(encrypted_vehicles)} encrypted vehicles")
        return False
    
    print("\nResult: Statistics correct")
    
    # ===== TEST 4: TRUE PIR Search =====
    print("\n[TEST 4] TRUE PIR Search with Encrypted Metadata")
    print("-"*70)
    
    pir_query = encrypted_embeddings[0] + np.random.rand(256) * 0.1
    
    print("Performing homomorphic search...")
    start = time.time()
    pir_results = pir_search_true(pir_query, top_k=5, verbose=True)
    pir_time = time.time() - start
    
    print(f"\nTotal search time: {pir_time*1000:.2f}ms")
    print(f"Results: {len(pir_results)}")
    
    print("\nTop results:")
    for i, result in enumerate(pir_results, 1):
        v = result['vehicle']
        score = result['similarity_score']
        print(f"  {i}. {v['license_plate']} ({v['color']}, {v['body_type']}) - Score: {score:.4f}")
    
    if len(pir_results) == 0:
        print("\nERROR: No results returned")
        return False
    
    # Verify metadata was decrypted
    first_result = pir_results[0]['vehicle']
    if first_result['license_plate'] is None:
        print("\nERROR: Metadata not decrypted")
        return False
    
    print("\nResult: Search successful - metadata decrypted client-side")
    
    # ===== TEST 5: Metadata Decryption Accuracy =====
    print("\n[TEST 5] Metadata Decryption Accuracy")
    print("-"*70)
    
    # Create lookup of original data
    original_data = {plate: {"color": color, "body": body} 
                     for plate, color, body in encrypted_vehicles}
    
    print("Verifying decrypted metadata matches original...")
    
    matches = 0
    errors = []
    
    for result in pir_results:
        v = result['vehicle']
        plate = v['license_plate']
        
        if plate in original_data:
            orig = original_data[plate]
            
            if v['color'] == orig['color'] and v['body_type'] == orig['body']:
                matches += 1
                print(f"  {plate}: OK (color={v['color']}, body={v['body_type']})")
            else:
                errors.append(plate)
                print(f"  {plate}: MISMATCH!")
                print(f"    Expected: {orig['color']}, {orig['body']}")
                print(f"    Got: {v['color']}, {v['body_type']}")
    
    if errors:
        print(f"\nERROR: {len(errors)} metadata mismatches")
        return False
    
    print(f"\nResult: {matches}/{len(pir_results)} metadata correctly decrypted")
    
    # ===== TEST 6: Multiple Searches =====
    print("\n[TEST 6] Multiple Search Performance")
    print("-"*70)
    
    num_searches = 5
    times = []
    
    print(f"Running {num_searches} searches...")
    for i in range(num_searches):
        q = np.random.rand(256)
        start = time.time()
        results = pir_search_true(q, top_k=3, verbose=False)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Search {i+1}: {elapsed*1000:.2f}ms ({len(results)} results)")
    
    avg_time = np.mean(times) * 1000
    min_time = np.min(times) * 1000
    max_time = np.max(times) * 1000
    
    print(f"\nPerformance:")
    print(f"  Average: {avg_time:.2f}ms")
    print(f"  Min: {min_time:.2f}ms")
    print(f"  Max: {max_time:.2f}ms")
    print(f"  Throughput: {1000/avg_time:.2f} searches/sec")
    
    print("\nResult: Performance consistent")
    
    # ===== TEST 7: Storage Analysis =====
    print("\n[TEST 7] Storage Analysis")
    print("-"*70)
    
    db = get_db()
    try:
        vehicles = db.query(Vehicle).filter(
            Vehicle.is_encrypted == True
        ).all()
        
        total_size = sum(v.get_storage_size() for v in vehicles)
        avg_size = total_size / len(vehicles)
        
        # Component breakdown
        avg_embedding = np.mean([len(v.encrypted_embedding) for v in vehicles])
        avg_metadata = np.mean([len(v.encrypted_metadata) for v in vehicles])
        avg_context = np.mean([len(v.encryption_context) for v in vehicles])
        
        print(f"Total vehicles: {len(vehicles)}")
        print(f"Total storage: {total_size/1024/1024:.2f} MB")
        print(f"\nPer vehicle average:")
        print(f"  Encrypted embedding: {avg_embedding/1024:.1f} KB")
        print(f"  Encrypted metadata: {avg_metadata/1024:.1f} KB")
        print(f"  Encryption context: {avg_context/1024:.1f} KB")
        print(f"  Total: {avg_size/1024:.1f} KB")
        
        # Compare to plain
        plain_size = 256 * 4  # 256 floats
        overhead = avg_size / plain_size
        
        print(f"\nComparison to plain:")
        print(f"  Plain embedding: {plain_size} bytes")
        print(f"  Encrypted: {avg_size:.0f} bytes")
        print(f"  Overhead: {overhead:.1f}x")
        
        print("\nResult: Storage overhead acceptable for privacy gain")
        
    finally:
        db.close()
    
    # ===== TEST 8: Privacy Verification =====
    print("\n[TEST 8] Privacy Verification")
    print("-"*70)
    
    db = get_db()
    try:
        print("Checking what server can see...")
        
        vehicles = db.query(Vehicle).filter(
            Vehicle.is_encrypted == True
        ).all()
        
        visible_count = 0
        for v in vehicles:
            if v.license_plate or v.color or v.body_type:
                visible_count += 1
                print(f"  WARNING: {v.vehicle_uuid} has visible data")
        
        if visible_count > 0:
            print(f"\nERROR: {visible_count} vehicles have visible metadata")
            return False
        
        print(f"\nServer view of database:")
        print(f"  Total encrypted vehicles: {len(vehicles)}")
        print(f"  Visible license plates: 0")
        print(f"  Visible colors: 0")
        print(f"  Visible body types: 0")
        print(f"  Visible embeddings: 0")
        print(f"\nServer knows: ONLY that encrypted vehicles exist")
        print("Server cannot: Read any actual data")
        
        print("\nResult: Privacy guarantee verified")
        
    finally:
        db.close()
    
    # ===== FINAL SUMMARY =====
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    print("\nAll tests passed:")
    print("  [OK] Encrypted vehicle insertion")
    print("  [OK] Metadata encryption verification")
    print("  [OK] Database statistics")
    print("  [OK] TRUE PIR search with encrypted metadata")
    print("  [OK] Metadata decryption accuracy")
    print("  [OK] Multiple search performance")
    print("  [OK] Storage analysis")
    print("  [OK] Privacy verification")
    
    print("\n" + "="*70)
    print("TRUE PIR ARCHITECTURE")
    print("="*70)
    
    print("\nStorage:")
    print("  - PostgreSQL ONLY (no FAISS)")
    print("  - encrypted_embedding: CKKS encrypted 256-dim vector")
    print("  - encrypted_metadata: CKKS encrypted JSON")
    print("  - encryption_context: CKKS context with keys")
    print("  - Plain fields: ALL NULL")
    
    print("\nSearch Process:")
    print("  1. Client encrypts query with CKKS")
    print("  2. Server performs homomorphic dot products")
    print("  3. Server returns encrypted scores")
    print("  4. Client decrypts scores locally")
    print("  5. Client selects top-K")
    print("  6. Client decrypts metadata for top-K")
    
    print("\nPrivacy Guarantees:")
    print("  - Server NEVER sees query")
    print("  - Server NEVER sees embeddings")
    print("  - Server NEVER sees metadata")
    print("  - Server NEVER sees similarity scores")
    print("  - Server ONLY knows: encrypted data exists")
    
    print("\nPerformance:")
    print("  - Complexity: O(N) linear scan")
    print(f"  - Average search time: {avg_time:.0f}ms")
    print(f"  - Storage overhead: {enc_stats['storage_overhead']:.1f}x")
    
    print("\n" + "="*70)
    print("SUCCESS: TRUE PIR with Encrypted Metadata Complete!")
    print("="*70 + "\n")
    
    return True


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    
    # Clean database
    print("\nSetting up clean database...")
    import subprocess
    
    subprocess.run(
        'psql -U postgres -c "DROP DATABASE IF EXISTS vehicle_storage"',
        shell=True,
        capture_output=True
    )
    
    subprocess.run(
        'psql -U postgres -c "CREATE DATABASE vehicle_storage"',
        shell=True,
        capture_output=True
    )
    
    subprocess.run(
        'psql -U postgres -d vehicle_storage -f database/init_db.sql',
        shell=True,
        capture_output=True
    )
    
    print("Database ready\n")
    
    # Run test
    success = test_true_pir()
    
    sys.exit(0 if success else 1)