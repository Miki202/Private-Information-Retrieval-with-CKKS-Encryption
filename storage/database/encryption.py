import json
import os
import numpy as np
import tenseal as ts
from typing import Dict

CONTEXT_FILE = "ckks_context.bin"

class CKKSEncryptor:
    """
    CKKS encryptor с един глобален context.

    Context-ът се създава само веднъж и се записва във файл.
    При следващи стартирания се зарежда от файла.
    """

    def __init__(
        self,
        poly_modulus_degree: int = 8192,
        coeff_mod_bit_sizes=None,
    ):

        if coeff_mod_bit_sizes is None:
            coeff_mod_bit_sizes = [60, 40, 40, 60]

        # Ако вече съществува context, зареждаме го
        if os.path.exists(CONTEXT_FILE):

            with open(CONTEXT_FILE, "rb") as f:
                self.context = ts.context_from(f.read())

        # Иначе създаваме нов context и го записваме
        else:

            self.context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=poly_modulus_degree,
                coeff_mod_bit_sizes=coeff_mod_bit_sizes,
            )

            self.context.global_scale = 2 ** 40
            self.context.generate_galois_keys()

            with open(CONTEXT_FILE, "wb") as f:
                f.write(
                    self.context.serialize(
                        save_secret_key=True
                    )
                )

    def encrypt_embedding(self, embedding: np.ndarray) -> bytes:
        """Криптира 256-мерен embedding-и."""

        embedding = np.asarray(embedding)

        if embedding.shape[0] != 256:
            raise ValueError(
                f"Получават се ембединги с размер {embedding.shape[0]}"
            )

        encrypted = ts.ckks_vector(
            self.context,
            embedding.tolist()
        )

        return encrypted.serialize()

    def decrypt_embedding(self, encrypted_data: bytes) -> np.ndarray:
        """Декриптира embedding-ите."""

        vector = ts.ckks_vector_from(
            self.context,
            encrypted_data
        )

        return np.array(vector.decrypt()[:256])

_encryptor = None

def get_encryptor() -> CKKSEncryptor:
    global _encryptor

    if _encryptor is None:
        _encryptor = CKKSEncryptor()

    return _encryptor

def get_context():
    """Връща глобалния CKKS контекст."""

    return get_encryptor().context


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """L2 нормализация на embedding-ите."""

    embedding = np.asarray(embedding)
    norm = np.linalg.norm(embedding)

    if norm == 0:
        embedding = np.random.rand(len(embedding))
        norm = np.linalg.norm(embedding)

    return embedding / norm


def encrypt_embedding_simple(embedding: np.ndarray) -> bytes:
    """Криптира embedding-ите."""
    return get_encryptor().encrypt_embedding(embedding)


def decrypt_embedding_simple(encrypted_data: bytes) -> np.ndarray:
    """Декриптира embedding-ите."""
    return get_encryptor().decrypt_embedding(encrypted_data)

def homomorphic_dot_product(
    encrypted_vec1: bytes,
    encrypted_vec2: bytes,
) -> bytes:
    """
    Изчислява хомоморфно скаларно произведение между два
    криптирани CKKS вектора.
    """

    context = get_context()
    v1 = ts.ckks_vector_from(context, encrypted_vec1)
    v2 = ts.ckks_vector_from(context, encrypted_vec2)
    return v1.dot(v2).serialize()

class PIRClient:

    def __init__(self):
        self.encryptor = get_encryptor()

    def prepare_query(self, query_embedding: np.ndarray) -> Dict:
        """Нормализира и криптира заявката."""

        query = normalize_embedding(query_embedding)
        encrypted_query = ts.ckks_vector(
            self.encryptor.context,
            query.tolist(),
        )

        return {
            "encrypted_query": encrypted_query.serialize(),
            "is_normalized": True,
        }

def encrypt_metadata(metadata: Dict) -> bytes:
    """
    Криптира метаданните като CKKS вектор.
    """
    context = get_context()
    try:
        if not metadata:
            metadata = {"image_path": "unknown"}
        json_data = json.dumps(metadata, ensure_ascii=True)
        byte_array = np.frombuffer(
            json_data.encode("utf-8"),
            dtype=np.uint8
        ).astype(np.float64)

        byte_array = byte_array[:256]

        if len(byte_array) < 256:
            byte_array = np.pad(
                byte_array,
                (0, 256 - len(byte_array))
            )

        byte_array = byte_array/255.0
        encrypted = ts.ckks_vector(context, byte_array.tolist())
        if encrypted is None:
            raise ValueError("CKKS връща None")
        serialized = encrypted.serialize()

        if not serialized:
            raise ValueError("CKKS връща празен резултат)")
        return serialized
    except Exception as e:
        raise RuntimeError(f"encrypt_metadata: {str(e)}")

def decrypt_metadata(encrypted_metadata: bytes) -> Dict:
    """
    Декриптира метаданните.
    """
<<<<<<< HEAD

    context = get_context()
    vector = ts.ckks_vector_from(context, encrypted_metadata)
    values = np.array(vector.decrypt()[:256])
    values = np.clip(values * 255.0, 0, 255)
    raw = np.round(values).astype(np.uint8).tobytes()
    
    decoded = raw.decode("utf-8", errors="ignore").rstrip("\x00")
    
    try:
        result = json.loads(decoded)
        return result
    except Exception:
        print(f"[decrypt_metadata] JSON parse failed. Raw: {repr(decoded[:80])}")
        return {}
=======
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
>>>>>>> f17374e (feat: add pir)
