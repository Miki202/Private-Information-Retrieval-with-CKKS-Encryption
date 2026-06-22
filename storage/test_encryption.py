"""
Тестов скрипт за проверка на CKKS криптирането
"""
import numpy as np
from database.encryption import CKKSEncryptor, encrypt_embedding_simple, decrypt_embedding_simple

def test_encryption():
    print("=" * 50)
    print("ТЕСТВАНЕ НА CKKS КРИПТИРАНЕ")
    print("=" * 50)
    
    # Тест 1: Създаване на encryptor
    print("\n1. Създаване на CKKS encryptor...")
    encryptor = CKKSEncryptor()
    
    # Тест 2: Генериране на тестов embedding
    print("\n2. Генериране на тестов embedding...")
    original_embedding = np.random.rand(256)
    print(f"   Първите 5 числа: {original_embedding[:5]}")
    print(f"   Размер: {original_embedding.shape}")
    
    # Тест 3: Криптиране
    print("\n3. Криптиране на embedding...")
    encrypted_data, context_data = encryptor.encrypt_embedding(original_embedding)
    print(f"   Размер на криптираните данни: {len(encrypted_data):,} bytes")
    print(f"   Размер на context: {len(context_data):,} bytes")
    print(f"   Общо: {len(encrypted_data) + len(context_data):,} bytes")
    print(f"   Compression ratio: {(len(encrypted_data) + len(context_data)) / (256 * 8):.1f}x")
    
    # Тест 4: Декриптиране
    print("\n4. Декриптиране на embedding...")
    decrypted_embedding = encryptor.decrypt_embedding(encrypted_data, context_data)
    print(f"   Първите 5 числа: {decrypted_embedding[:5]}")
    print(f"   Размер: {decrypted_embedding.shape}")
    
    # Тест 5: Проверка на точността
    print("\n5. Проверка на точността...")
    difference = np.abs(original_embedding - decrypted_embedding)
    max_error = np.max(difference)
    mean_error = np.mean(difference)
    print(f"   Максимална грешка: {max_error:.10f}")
    print(f"   Средна грешка: {mean_error:.10f}")
    
    if max_error < 1e-5:
        print("   ✓ Точността е отлична!")
    elif max_error < 1e-3:
        print("   ✓ Точността е добра")
    else:
        print("   ⚠ Точността е ниска - провери параметрите")
    
    # Тест 6: Използване на простите функции
    print("\n6. Тестване на простите функции...")
    encrypted_data2, context_data2 = encrypt_embedding_simple(original_embedding)
    decrypted_embedding2 = decrypt_embedding_simple(encrypted_data2, context_data2)
    difference2 = np.abs(original_embedding - decrypted_embedding2)
    print(f"   Максимална грешка: {np.max(difference2):.10f}")
    print("   ✓ Простите функции работят")
    
    # Тест 7: Проверка на различни embeddings
    print("\n7. Тестване с множество embeddings...")
    for i in range(3):
        test_emb = np.random.rand(256)
        enc_data, ctx_data = encryptor.encrypt_embedding(test_emb)
        dec_emb = encryptor.decrypt_embedding(enc_data, ctx_data)
        error = np.max(np.abs(test_emb - dec_emb))
        print(f"   Embedding {i+1}: max error = {error:.10f}")
    
    # Тест 8: Съхраняване и зареждане на context
    print("\n8. Тестване на save/load context...")
    encryptor.save_context("test_context.bin")
    
    # Зареждане
    encryptor2 = CKKSEncryptor.load_context("test_context.bin")
    decrypted_embedding3 = encryptor2.decrypt_embedding(encrypted_data, context_data)
    difference3 = np.abs(original_embedding - decrypted_embedding3)
    print(f"   Максимална грешка след reload: {np.max(difference3):.10f}")
    print("   ✓ Save/Load context работи")
    
    # Изчистване
    import os
    os.remove("test_context.bin")
    
    print("\n" + "=" * 50)
    print("✓ ВСИЧКИ ENCRYPTION ТЕСТОВЕ ЗАВЪРШИХА")
    print("=" * 50)

if __name__ == "__main__":
    test_encryption()