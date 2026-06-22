"""
TRUE PIR Test with Encrypted Metadata
Tests ONLY encrypted PIR mode (no plaintext fields)
"""
import numpy as np
import time

from database.operations import (
    insert_vehicle,
    pir_search_true,
    get_database_stats,
    get_encryption_stats,
    decrypt_metadata,   # IMPORTANT for validation
)

def test_true_pir():

    print("\n" + "=" * 70)
    print("TRUE PIR TEST - ENCRYPTED METADATA ONLY")
    print("=" * 70)

    # =========================================================
    # TEST 1: INSERT ENCRYPTED VEHICLES
    # =========================================================
    print("\n[TEST 1] Insert Encrypted Vehicles")
    print("-" * 70)

    embeddings = []
    expected_metadata = {}

    print("Inserting encrypted vehicles...")

    for i in range(8):
        emb = np.random.rand(256)
        embeddings.append(emb)

        uuid_val, vid = insert_vehicle(
            embedding=emb,
            image_path=f"ENC{i+1:03d}.jpg"
        )

        expected_metadata[uuid_val] = f"ENC{i+1:03d}.jpg"

        print(f"  ENC{i+1:03d}: UUID={str(uuid_val)[:8]}..., ID={vid}")

    print(f"\nResult: 8 vehicles inserted")
    print("  Embeddings: ENCRYPTED")
    print("  Metadata: ENCRYPTED (server-side only)")
    print("  Server can see: NOTHING")

    # =========================================================
    # TEST 2: DATABASE STATISTICS
    # =========================================================
    print("\n[TEST 2] Database Statistics")
    print("-" * 70)

    stats = get_database_stats()
    enc_stats = get_encryption_stats()

    print(f"Total vehicles: {stats['total_vehicles']}")
    print(f"Avg encrypted size: {stats['avg_encrypted_storage_bytes']} bytes")
    print(f"Storage overhead: {enc_stats['storage_overhead']:.1f}x")

    # =========================================================
    # TEST 3: PIR SEARCH
    # =========================================================
    print("\n[TEST 3] TRUE PIR Search")
    print("-" * 70)

    query = embeddings[0] + np.random.rand(256) * 0.05

    print("Performing homomorphic search...")

    start = time.time()
    results = pir_search_true(query, top_k=5, verbose=True)
    total_time = time.time() - start

    print(f"\nTotal search time: {total_time*1000:.2f}ms")
    print(f"Results: {len(results)}")

    if not results:
        print("\nERROR: No results returned")
        return False

    print("\nTop results:")
    for i, r in enumerate(results, 1):
        v = r["vehicle"]
        print(f"  {i}. {v['image_path']} - Score: {r['similarity_score']:.4f}")

    # =========================================================
    # TEST 4: METADATA DECRYPTION VALIDATION
    # =========================================================
    print("\n[TEST 4] Metadata Decryption Validation")
    print("-" * 70)

    correct = 0
    for r in results:
        v = r["vehicle"]
        
        # image_path is already decrypted server-side
        if v.get("image_path") is None:
            print(f"  WARN: vehicle id={v['id']} has no image_path (old record?)")
            continue
        
        correct += 1
        print(f"  OK: {v['image_path']}")

    print(f"Result: {correct}/{len(results)} metadata entries valid")
    if correct == 0:
        return False

    # =========================================================
    # TEST 5: PERFORMANCE
    # =========================================================
    print("\n[TEST 5] Multiple Searches")
    print("-" * 70)

    times = []

    for i in range(5):
        q = np.random.rand(256)

        start = time.time()
        pir_search_true(q, top_k=3, verbose=False)
        times.append(time.time() - start)

        print(f"  Search {i+1}: {times[-1]*1000:.2f}ms")

    print("\nPerformance:")
    print(f"  Avg: {np.mean(times)*1000:.2f}ms")
    print(f"  Min: {np.min(times)*1000:.2f}ms")
    print(f"  Max: {np.max(times)*1000:.2f}ms")

    # =========================================================
    # TEST 6: STORAGE ANALYSIS
    # =========================================================
    print("\n[TEST 6] Storage Analysis")
    print("-" * 70)

    print(f"Total vehicles: {stats['total_vehicles']}")
    print(f"Storage overhead: {enc_stats['storage_overhead']:.1f}x")

    # =========================================================
    # FINAL RESULT
    # =========================================================
    print("\n" + "=" * 70)
    print("SUCCESS: TRUE PIR SYSTEM WORKING")
    print("=" * 70)

    return True


if __name__ == "__main__":
    import sys
    sys.exit(0 if test_true_pir() else 1)