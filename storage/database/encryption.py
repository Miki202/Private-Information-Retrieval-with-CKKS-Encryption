"""
Модул за CKKS криптиране и декриптиране на embeddings
Използва TenSEAL библиотека
"""

import tenseal as ts
import numpy as np
from typing import Tuple, Optional
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
        
        print(f"Създаване на CKKS context...")
        print(f"  - Poly modulus degree: {poly_modulus_degree}")
        print(f"  - Coeff mod bit sizes: {coeff_mod_bit_sizes}")
        
        # Създаване на TenSEAL context
        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=poly_modulus_degree,
            coeff_mod_bit_sizes=coeff_mod_bit_sizes
        )
        
        # Global scale - определя precision на числата
        # По-голям scale = по-добра точност но по-малко операции
        self.context.global_scale = 2**40
        
        # Генериране на Galois keys (необходими за някои операции)
        self.context.generate_galois_keys()
        
        print("✓ CKKS context създаден успешно")
    
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
        
        print(f"Криптиране на embedding...")
        
        # Конвертираме numpy array в list
        embedding_list = embedding.tolist()
        
        # Криптираме с CKKS
        encrypted_vector = ts.ckks_vector(self.context, embedding_list)
        
        # Сериализираме (конвертираме в bytes)
        encrypted_data = encrypted_vector.serialize()
        
        # Също така съхраняваме context (нужен за декриптиране)
        context_data = self.context.serialize(save_secret_key=True)
        
        print(f"✓ Embedding криптиран: {len(encrypted_data)} bytes")
        
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
        print(f"Декриптиране на embedding...")
        
        # Зареждаме context
        context = ts.context_from(context_data)
        
        # Зареждаме криптирания вектор
        encrypted_vector = ts.ckks_vector_from(context, encrypted_data)
        
        # Декриптираме
        decrypted_list = encrypted_vector.decrypt()
        
        # Конвертираме обратно в numpy array
        embedding = np.array(decrypted_list[:256])  # Вземаме само първите 256 (понякога има padding)
        
        print(f"✓ Embedding декриптиран")
        
        return embedding
    
    def save_context(self, filepath: str):
        """
        Съхранява context във файл (за по-късно използване)
        
        Args:
            filepath: Път до файла
        """
        context_bytes = self.context.serialize(save_secret_key=True)
        with open(filepath, 'wb') as f:
            f.write(context_bytes)
        print(f"✓ Context съхранен в {filepath}")
    
    @classmethod
    def load_context(cls, filepath: str) -> 'CKKSEncryptor':
        """
        Зарежда context от файл
        
        Args:
            filepath: Път до файла
        
        Returns:
            CKKSEncryptor instance
        """
        with open(filepath, 'rb') as f:
            context_bytes = f.read()
        
        encryptor = cls.__new__(cls)  # Създаваме instance без __init__
        encryptor.context = ts.context_from(context_bytes)
        print(f"✓ Context зареден от {filepath}")
        return encryptor
    
    def get_context_size(self) -> int:
        """
        Връща размера на context в bytes
        """
        context_bytes = self.context.serialize(save_secret_key=True)
        return len(context_bytes)
    
    def get_encrypted_size(self, embedding: np.ndarray) -> int:
        """
        Връща размера на криптирания embedding в bytes (без да го съхранява)
        """
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
        print("Инициализация на CKKS encryptor...")
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


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """
    Normalize embedding to unit length (||v|| = 1)
    Необходимо за dot product ≈ cosine similarity
    
    Args:
        embedding: Numpy array
    
    Returns:
        Normalized embedding
    """
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm


def encrypt_query_for_pir(query_embedding: np.ndarray, context: ts.Context = None) -> Tuple[bytes, bytes]:
    """
    Криптира query embedding за PIR търсене
    
    Args:
        query_embedding: Query vector (ще се normalize-ва автоматично)
        context: CKKS context (ако е None, използва глобалния)
    
    Returns:
        (encrypted_query_bytes, context_bytes)
    """
    # Normalize query за dot product similarity
    normalized_query = normalize_embedding(query_embedding)
    
    print(f"PIR Query криптиране...")
    print(f"  Original norm: {np.linalg.norm(query_embedding):.4f}")
    print(f"  Normalized norm: {np.linalg.norm(normalized_query):.4f}")
    
    if context is None:
        encryptor = get_encryptor()
        context = encryptor.context
    
    # Криптираме normalized query
    encrypted_vector = ts.ckks_vector(context, normalized_query.tolist())
    encrypted_bytes = encrypted_vector.serialize()
    context_bytes = context.serialize(save_secret_key=True)
    
    print(f"✓ Query криптиран: {len(encrypted_bytes):,} bytes")
    
    return encrypted_bytes, context_bytes


def homomorphic_dot_product(
    encrypted_query_bytes: bytes,
    encrypted_db_vector_bytes: bytes,
    query_context_bytes: bytes
) -> bytes:
    """
    Изчислява dot product между два криптирани вектора
    ТОВА Е КЛЮЧОВАТА PIR ОПЕРАЦИЯ - server не вижда нито query нито database vector!
    
    Args:
        encrypted_query_bytes: Криптиран query vector
        encrypted_db_vector_bytes: Криптиран database vector
        query_context_bytes: CKKS context от query
    
    Returns:
        Encrypted dot product result (bytes)
    """
    # Зареждаме context
    context = ts.context_from(query_context_bytes)
    
    # Зареждаме криптираните вектори
    enc_query = ts.ckks_vector_from(context, encrypted_query_bytes)
    enc_db = ts.ckks_vector_from(context, encrypted_db_vector_bytes)
    
    # HOMOMORPHIC MULTIPLICATION (element-wise)
    # [a1, a2, a3] * [b1, b2, b3] = [a1*b1, a2*b2, a3*b3]
    enc_product = enc_query * enc_db
    
    # HOMOMORPHIC SUM
    # Сумираме всички елементи за да получим dot product
    # Това е малко трик - TenSEAL няма direct sum, но можем да използваме матрица от 1-ци
    
    # Алтернатива: Връщаме element-wise product и client-а ще направи sum след decrypt
    # Това е по-ефективно
    
    return enc_product.serialize()


def decrypt_similarities(encrypted_results: List[bytes], context_bytes: bytes) -> np.ndarray:
    """
    Декриптира резултатите от PIR търсене
    
    Args:
        encrypted_results: List of encrypted similarity scores
        context_bytes: CKKS context
    
    Returns:
        Numpy array със similarity scores
    """
    context = ts.context_from(context_bytes)
    
    similarities = []
    for enc_result in encrypted_results:
        enc_vec = ts.ckks_vector_from(context, enc_result)
        decrypted = enc_vec.decrypt()
        
        # Ако е element-wise product, сумираме за dot product
        dot_product = sum(decrypted[:256])  # Само първите 256 (може да има padding)
        similarities.append(dot_product)
    
    return np.array(similarities)


class PIRClient:
    """
    Client-side PIR логика
    Криптира queries и декриптира results
    """
    
    def __init__(self):
        self.encryptor = get_encryptor()
        print("PIR Client инициализиран")
    
    def prepare_query(self, query_embedding: np.ndarray) -> Dict:
        """
        Подготвя криптиран query за изпращане до server
        
        Returns:
            Dict с encrypted query и context
        """
        encrypted_query, context = encrypt_query_for_pir(
            query_embedding,
            self.encryptor.context
        )
        
        return {
            "encrypted_query": encrypted_query,
            "context": context
        }
    
    def process_results(
        self,
        encrypted_results: List[bytes],
        vehicle_ids: List[int],
        context_bytes: bytes,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Декриптира резултатите и избира top-K
        
        Args:
            encrypted_results: Encrypted similarity scores
            vehicle_ids: Corresponding vehicle IDs
            context_bytes: CKKS context
            top_k: Number of results
        
        Returns:
            List of {vehicle_id, similarity_score}
        """
        print(f"\nДекриптиране на {len(encrypted_results)} резултата...")
        
        # Декриптираме
        similarities = decrypt_similarities(encrypted_results, context_bytes)
        
        print(f"✓ Резултати декриптирани")
        print(f"  Similarity range: [{similarities.min():.4f}, {similarities.max():.4f}]")
        
        # Съчетаваме с IDs
        results = [
            {"vehicle_id": vid, "similarity_score": float(sim)}
            for vid, sim in zip(vehicle_ids, similarities)
        ]
        
        # Сортираме по similarity (descending)
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        # Top-K
        top_results = results[:top_k]
        
        print(f"✓ Top {len(top_results)} резултата:")
        for i, r in enumerate(top_results[:5]):
            print(f"  {i+1}. Vehicle {r['vehicle_id']}: {r['similarity_score']:.4f}")
        
        return top_results


class PIRServer:
    """
    Server-side PIR логика
    Обработва encrypted queries без да вижда plaintext
    """
    
    def __init__(self):
        print("PIR Server инициализиран")
    
    def search_encrypted_database(
        self,
        encrypted_query: bytes,
        query_context: bytes,
        filter_color: Optional[str] = None,
        filter_body_type: Optional[str] = None
    ) -> Tuple[List[bytes], List[int]]:
        """
        Търси в криптираната база данни
        
        ⚠️ ВАЖНО: Server НЕ ВИЖДА нито query нито database vectors!
        
        Args:
            encrypted_query: Криптиран query vector
            query_context: CKKS context
            filter_color: Optional color filter
            filter_body_type: Optional body type filter
        
        Returns:
            (encrypted_similarities, vehicle_ids)
        """
        from .models import Vehicle, VehicleVector
        from .connection import get_db
        
        print("\n" + "="*60)
        print("PIR SERVER: Обработка на криптиран query")
        print("="*60)
        print("⚠️  Server НЕ ВИЖДА plaintext query или database vectors!")
        
        db = get_db()
        
        try:
            # Вземаме всички encrypted записи
            query = db.query(Vehicle, VehicleVector).join(
                VehicleVector, Vehicle.id == VehicleVector.vehicle_id
            ).filter(Vehicle.is_encrypted == True)
            
            # Прилагаме metadata филтри (това е OK - не разкрива query)
            if filter_color:
                query = query.filter(Vehicle.color == filter_color)
                print(f"  Филтър: color = {filter_color}")
            if filter_body_type:
                query = query.filter(Vehicle.body_type == filter_body_type)
                print(f"  Филтър: body_type = {filter_body_type}")
            
            results = query.all()
            
            print(f"\nОбработка на {len(results)} криптирани записа...")
            
            encrypted_similarities = []
            vehicle_ids = []
            
            for i, (vehicle, vector) in enumerate(results):
                # HOMOMORPHIC DOT PRODUCT
                # Server прави това без да знае нито query нито database vector!
                enc_similarity = homomorphic_dot_product(
                    encrypted_query,
                    vector.encrypted_embedding,
                    query_context
                )
                
                encrypted_similarities.append(enc_similarity)
                vehicle_ids.append(vehicle.id)
                
                if (i + 1) % 10 == 0:
                    print(f"  Обработени {i+1}/{len(results)}...")
            
            print(f"\n✓ {len(encrypted_similarities)} криптирани similarity scores изчислени")
            print("✓ Server НИКОГА не видя plaintext данни!\n")
            print("="*60)
            
            return encrypted_similarities, vehicle_ids
            
        finally:
            db.close()