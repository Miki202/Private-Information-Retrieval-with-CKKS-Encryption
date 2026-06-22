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
