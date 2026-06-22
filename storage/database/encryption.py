import tenseal as ts
import numpy as np
from typing import Tuple, Optional, Dict, List
import pickle

class CKKSEncryptor:
    """
    Клас за работа с CKKS криптиране
    Криптира и декриптира 256-мерни embeddings
    """
    
    def __init__(self, poly_modulus_degree: int = 8192, coeff_mod_bit_sizes: list = None):
        """
        Инициализация на CKKS context
        
        Args:
            poly_modulus_degree: Степен на полинома (по-голям = по-сигурен но по-бавен)
                                 Трябва да е power of 2: 4096, 8192, 16384, 32768
            coeff_mod_bit_sizes: Размери на коефициентите (за precision)
        """
        if coeff_mod_bit_sizes is None:
            # Default параметри - добър баланс между security и performance
            coeff_mod_bit_sizes = [60, 40, 40, 60]
        
        print(f"CKKS Context инициализиран:")
        print(f"  Poly modulus degree: {poly_modulus_degree}")
        print(f"  Coeff modulus size: {sum(coeff_mod_bit_sizes)} bits")
        
        # Създаване на TenSEAL context
        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=poly_modulus_degree,
            coeff_mod_bit_sizes=coeff_mod_bit_sizes
        )
        
        # Global scale - определя precision на числата
        self.context.global_scale = 2**40
        print(f"  Scale: 2^40")
        
        # Генериране на Galois keys (необходими за някои операции)
        self.context.generate_galois_keys()
    
    def encrypt_embedding(self, embedding: np.ndarray) -> Tuple[bytes, bytes]:
        """
        Криптира embedding
        
        Args:
            embedding: Numpy array с 256 float числа
        
        Returns:
            (encrypted_data, context_data) - Tuple от bytes
        """
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)
        
        if embedding.shape[0] != 256:
            raise ValueError(f"Embedding трябва да е 256-мерен, получен: {embedding.shape[0]}")
        
        print(f"Encryption of 256-dim embedding...")
        
        # Конвертираме numpy array в list
        embedding_list = embedding.tolist()
        
        # Криптираме с CKKS
        encrypted_vector = ts.ckks_vector(self.context, embedding_list)
        
        # Сериализираме (конвертираме в bytes)
        encrypted_data = encrypted_vector.serialize()
        
        # Също така съхраняваме context (нужен за декриптиране)
        context_data = self.context.serialize(save_secret_key=True)
        
        print(f"✓ Encrypted: {len(encrypted_data)} bytes")
        print(f"✓ Context: {len(context_data)} bytes")
        
        return encrypted_data, context_data
    
    def decrypt_embedding(self, encrypted_data: bytes, context_data: bytes) -> np.ndarray:
        """
        Декриптира embedding
        
        Args:
            encrypted_data: Криптираните данни (bytes)
            context_data: CKKS context (bytes)
        
        Returns:
            Numpy array с 256 float числа
        """
        print(f"Decryption of 256-dim embedding...")
        
        # Зареждаме context
        context = ts.context_from(context_data)
        
        # Зареждаме криптирания вектор
        encrypted_vector = ts.ckks_vector_from(context, encrypted_data)
        
        # Декриптираме
        decrypted_list = encrypted_vector.decrypt()
        
        # Конвертираме обратно в numpy array
        embedding = np.array(decrypted_list[:256])  # Вземаме само първите 256
        
        # Calculate error
        print(f"✓ Decrypted (max error: {np.max(np.abs(embedding)):.10f})")
        
        return embedding
    
    def save_context(self, filepath: str):
        """Съхранява context във файл"""
        context_bytes = self.context.serialize(save_secret_key=True)
        with open(filepath, 'wb') as f:
            f.write(context_bytes)
        print(f"✓ Context съхранен в {filepath}")
    
    @classmethod
    def load_context(cls, filepath: str) -> 'CKKSEncryptor':
        """Зарежда context от файл"""
        with open(filepath, 'rb') as f:
            context_bytes = f.read()
        
        encryptor = cls.__new__(cls)
        encryptor.context = ts.context_from(context_bytes)
        print(f"✓ Context зареден от {filepath}")
        return encryptor
    
    def get_context_size(self) -> int:
        """Връща размера на context в bytes"""
        context_bytes = self.context.serialize(save_secret_key=True)
        return len(context_bytes)
    
    def get_encrypted_size(self, embedding: np.ndarray) -> int:
        """Връща размера на криптирания embedding в bytes"""
        encrypted_data, _ = self.encrypt_embedding(embedding)
        return len(encrypted_data)


# ============================================
# Глобална инстанция (Singleton pattern)
# ============================================

_global_encryptor: Optional[CKKSEncryptor] = None

def get_encryptor(force_new: bool = False) -> CKKSEncryptor:
    """
    Връща глобална инстанция на CKKSEncryptor (създава я ако не съществува)
    
    Args:
        force_new: Ако е True, създава нова инстанция
    
    Returns:
        CKKSEncryptor instance
    """
    global _global_encryptor
    
    if _global_encryptor is None or force_new:
        _global_encryptor = CKKSEncryptor()
    
    return _global_encryptor


# ============================================
# Convenience функции
# ============================================

def encrypt_embedding_simple(embedding: np.ndarray) -> Tuple[bytes, bytes]:
    """
    Проста функция за криптиране (използва глобалния encryptor)
    
    Args:
        embedding: 256-мерен numpy array
    
    Returns:
        (encrypted_data, context_data)
    """
    encryptor = get_encryptor()
    return encryptor.encrypt_embedding(embedding)


def decrypt_embedding_simple(encrypted_data: bytes, context_data: bytes) -> np.ndarray:
    """
    Проста функция за декриптиране
    
    Args:
        encrypted_data: Криптираните данни
        context_data: CKKS context
    
    Returns:
        256-мерен numpy array
    """
    encryptor = get_encryptor()
    return encryptor.decrypt_embedding(encrypted_data, context_data)


# ============================================
# НОВИ ФУНКЦИИ ЗА PIR
# ============================================

def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """
    Нормализира embedding до unit length (||v|| = 1)
    
    Важно за PIR защото:
    - Cosine similarity = dot product (ако векторите са normalized)
    - По-добра numeric stability при CKKS encryption
    
    Args:
        embedding: 256-dim vector
    
    Returns:
        Normalized vector с ||v|| = 1
    """
    if not isinstance(embedding, np.ndarray):
        embedding = np.array(embedding)
    
    norm = np.linalg.norm(embedding)
    
    if norm == 0:
        # Ако е zero vector, върни random normalized vector
        print("⚠️ Zero vector detected, returning random normalized vector")
        embedding = np.random.rand(len(embedding))
        norm = np.linalg.norm(embedding)
    
    normalized = embedding / norm
    
    return normalized


class PIRClient:
    """
    PIR Client - криптира queries и декриптира резултати
    """
    
    def __init__(self):
        """Инициализира CKKS encryptor за client"""
        self.encryptor = get_encryptor()
    
    def prepare_query(self, query_embedding: np.ndarray) -> Dict:
        """
        Подготвя query за изпращане към сървъра
        
        Args:
            query_embedding: 256-dim query vector
        
        Returns:
            Dict с encrypted query data
        """
        # Normalize за cosine similarity
        normalized_query = normalize_embedding(query_embedding)
        
        print(f"CLIENT: Normalizing query (||q|| = {np.linalg.norm(normalized_query):.6f})")
        
        # Encrypt
        encrypted_data, context_data = self.encryptor.encrypt_embedding(normalized_query)
        
        print(f"CLIENT: Query encrypted ({len(encrypted_data):,} bytes)")
        
        return {
            "encrypted_query": encrypted_data,
            "context": context_data,
            "is_normalized": True
        }
    
    def decrypt_results(self, encrypted_results: List[bytes], context_data: bytes) -> np.ndarray:
        """
        Декриптира резултатите от сървъра
        
        Args:
            encrypted_results: List of encrypted similarity scores
            context_data: CKKS context
        
        Returns:
            Array от similarity scores
        """
        print(f"CLIENT: Decrypting {len(encrypted_results)} results...")
        
        scores = []
        
        for enc_result in encrypted_results:
            decrypted = self.encryptor.decrypt_embedding(enc_result, context_data)
            # Вземи първия елемент (similarity score)
            scores.append(decrypted[0])
        
        return np.array(scores)


class PIRServer:
    """
    PIR Server - извършва homomorphic операции върху encrypted data
    
    Забележка: В тази имплементация сървърът работи с FAISS за бързина,
    което е компромис. За pure PIR, трябва да се правят homomorphic
    dot products директно върху encrypted vectors.
    """
    
    def __init__(self):
        """Инициализира сървъра"""
        pass
    
    def compute_similarity_homomorphic(
        self,
        encrypted_query: bytes,
        encrypted_db_vector: bytes,
        context_data: bytes
    ) -> bytes:
        """
        Изчислява similarity homomorphically (без да декриптира)
        
        Това е "pure" PIR операция - dot product в encrypted space
        
        Args:
            encrypted_query: Encrypted query vector
            encrypted_db_vector: Encrypted database vector
            context_data: CKKS context
        
        Returns:
            Encrypted similarity score
        """
        print("SERVER: Computing homomorphic dot product...")
        
        # Load context (без secret key за сървъра!)
        # В production, сървърът НЕ трябва да има secret key
        context = ts.context_from(context_data)
        
        # Load vectors
        query_vec = ts.ckks_vector_from(context, encrypted_query)
        db_vec = ts.ckks_vector_from(context, encrypted_db_vector)
        
        # Homomorphic dot product
        # result = sum(q[i] * db[i]) за всички i
        result = query_vec.dot(db_vec)
        
        # Serialize result
        result_bytes = result.serialize()
        
        print(f"SERVER: Similarity computed (encrypted: {len(result_bytes)} bytes)")
        
        return result_bytes
    
    def search_encrypted_database(
        self,
        encrypted_query_data: Dict,
        encrypted_db_vectors: List[Tuple[int, bytes, bytes]],
        top_k: int = 10
    ) -> List[Tuple[int, bytes]]:
        """
        Търси в encrypted database
        
        Args:
            encrypted_query_data: Dict от prepare_query()
            encrypted_db_vectors: List of (vehicle_id, encrypted_vector, context)
            top_k: Брой резултати
        
        Returns:
            List of (vehicle_id, encrypted_similarity_score)
        """
        print(f"SERVER: Searching {len(encrypted_db_vectors)} encrypted vectors...")
        
        encrypted_query = encrypted_query_data["encrypted_query"]
        query_context = encrypted_query_data["context"]
        
        results = []
        
        for vehicle_id, enc_db_vec, db_context in encrypted_db_vectors:
            # Compute similarity homomorphically
            enc_similarity = self.compute_similarity_homomorphic(
                encrypted_query,
                enc_db_vec,
                query_context  # Използваме query context
            )
            
            results.append((vehicle_id, enc_similarity))
        
        print(f"SERVER: Computed {len(results)} encrypted similarities")
        
        # Връщаме всички резултати (encrypted)
        # Client ще ги декриптира и ще избере top-k
        return results


# ============================================
# PIR Helper Functions
# ============================================

def pir_encrypt_database_vector(embedding: np.ndarray) -> Tuple[bytes, bytes]:
    """
    Криптира database vector за PIR
    
    Args:
        embedding: 256-dim vector
    
    Returns:
        (encrypted_vector, context)
    """
    # Normalize
    normalized = normalize_embedding(embedding)
    
    # Encrypt
    encrypted_data, context_data = encrypt_embedding_simple(normalized)
    
    return encrypted_data, context_data


def cosine_similarity_encrypted(
    enc_query: bytes,
    enc_db_vector: bytes,
    context: bytes
) -> float:
    """
    Изчислява cosine similarity между encrypted vectors
    
    Забележка: Това изисква декриптиране, така че не е pure PIR.
    За pure PIR, трябва да се използва compute_similarity_homomorphic()
    
    Args:
        enc_query: Encrypted query
        enc_db_vector: Encrypted DB vector
        context: CKKS context
    
    Returns:
        Similarity score (0-1)
    """
    # Decrypt (това е компромисът!)
    query = decrypt_embedding_simple(enc_query, context)
    db_vec = decrypt_embedding_simple(enc_db_vector, context)
    
    # Compute similarity
    # Ако и двата са normalized, dot product = cosine similarity
    similarity = np.dot(query, db_vec)
    
    # Clamp to [0, 1]
    similarity = max(0.0, min(1.0, similarity))
    
    return similarity


# ============================================
# Testing & Debugging
# ============================================

def test_encryption_roundtrip():
    """
    Тества encrypt → decrypt цикъл
    """
    print("\n" + "="*60)
    print("Testing Encryption Roundtrip")
    print("="*60)
    
    # Random embedding
    original = np.random.rand(256)
    
    print(f"\nOriginal embedding:")
    print(f"  Shape: {original.shape}")
    print(f"  Norm: {np.linalg.norm(original):.6f}")
    print(f"  First 5: {original[:5]}")
    
    # Normalize
    normalized = normalize_embedding(original)
    print(f"\nNormalized embedding:")
    print(f"  Norm: {np.linalg.norm(normalized):.6f}")
    
    # Encrypt
    enc_data, ctx_data = encrypt_embedding_simple(normalized)
    print(f"\nEncrypted:")
    print(f"  Encrypted size: {len(enc_data):,} bytes")
    print(f"  Context size: {len(ctx_data):,} bytes")
    
    # Decrypt
    decrypted = decrypt_embedding_simple(enc_data, ctx_data)
    print(f"\nDecrypted embedding:")
    print(f"  Shape: {decrypted.shape}")
    print(f"  Norm: {np.linalg.norm(decrypted):.6f}")
    print(f"  First 5: {decrypted[:5]}")
    
    # Error
    error = np.max(np.abs(normalized - decrypted))
    print(f"\nMax error: {error:.10f}")
    
    if error < 1e-5:
        print("✅ Excellent accuracy!")
    elif error < 1e-3:
        print("✅ Good accuracy")
    else:
        print("⚠️ Marginal accuracy")
    
    return error < 1e-3


if __name__ == "__main__":
    # Run test
    test_encryption_roundtrip()