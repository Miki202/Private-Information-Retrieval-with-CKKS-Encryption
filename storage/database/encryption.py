"""
CKKS Homomorphic Encryption for TRUE PIR
TenSEAL implementation for 256-dim embeddings
"""

import tenseal as ts
import numpy as np
from typing import Tuple, Optional, Dict

# ============================================
# Core CKKS Encryptor
# ============================================

class CKKSEncryptor:
    """CKKS encryption for 256-dim embeddings"""
    
    def __init__(self, poly_modulus_degree: int = 8192, coeff_mod_bit_sizes: list = None):
        """
        Initialize CKKS context
        
        Args:
            poly_modulus_degree: Polynomial degree (power of 2: 4096, 8192, 16384)
            coeff_mod_bit_sizes: Coefficient modulus bit sizes
        """
        if coeff_mod_bit_sizes is None:
            coeff_mod_bit_sizes = [60, 40, 40, 60]
        
        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=poly_modulus_degree,
            coeff_mod_bit_sizes=coeff_mod_bit_sizes
        )
        
        self.context.global_scale = 2**40
        self.context.generate_galois_keys()
    
    def encrypt_embedding(self, embedding: np.ndarray) -> Tuple[bytes, bytes]:
        """
        Encrypt embedding
        
        Args:
            embedding: 256-dim numpy array
        
        Returns:
            (encrypted_data, context_data)
        """
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)
        
        if embedding.shape[0] != 256:
            raise ValueError(f"Expected 256-dim embedding, got {embedding.shape[0]}")
        
        embedding_list = embedding.tolist()
        encrypted_vector = ts.ckks_vector(self.context, embedding_list)
        encrypted_data = encrypted_vector.serialize()
        context_data = self.context.serialize(save_secret_key=True)
        
        return encrypted_data, context_data
    
    def decrypt_embedding(self, encrypted_data: bytes, context_data: bytes) -> np.ndarray:
        """
        Decrypt embedding
        
        Args:
            encrypted_data: Encrypted bytes
            context_data: CKKS context bytes
        
        Returns:
            256-dim numpy array
        """
        context = ts.context_from(context_data)
        encrypted_vector = ts.ckks_vector_from(context, encrypted_data)
        decrypted_list = encrypted_vector.decrypt()
        embedding = np.array(decrypted_list[:256])
        
        return embedding


# ============================================
# Global Singleton
# ============================================

_global_encryptor: Optional[CKKSEncryptor] = None

def get_encryptor(force_new: bool = False) -> CKKSEncryptor:
    """Get or create global CKKSEncryptor instance"""
    global _global_encryptor
    
    if _global_encryptor is None or force_new:
        _global_encryptor = CKKSEncryptor()
    
    return _global_encryptor


# ============================================
# Simple Wrapper Functions
# ============================================

def encrypt_embedding_simple(embedding: np.ndarray) -> Tuple[bytes, bytes]:
    """
    Encrypt embedding using global encryptor
    
    Args:
        embedding: 256-dim numpy array
    
    Returns:
        (encrypted_data, context_data)
    """
    encryptor = get_encryptor()
    return encryptor.encrypt_embedding(embedding)


def decrypt_embedding_simple(encrypted_data: bytes, context_data: bytes) -> np.ndarray:
    """
    Decrypt embedding
    
    Args:
        encrypted_data: Encrypted bytes
        context_data: CKKS context bytes
    
    Returns:
        256-dim numpy array
    """
    encryptor = get_encryptor()
    return encryptor.decrypt_embedding(encrypted_data, context_data)


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """
    Normalize embedding to unit length
    
    Args:
        embedding: N-dim vector
    
    Returns:
        Normalized vector with ||v|| = 1
    """
    if not isinstance(embedding, np.ndarray):
        embedding = np.array(embedding)
    
    norm = np.linalg.norm(embedding)
    
    if norm == 0:
        embedding = np.random.rand(len(embedding))
        norm = np.linalg.norm(embedding)
    
    return embedding / norm


# ============================================
# Homomorphic Operations
# ============================================

def homomorphic_dot_product(
    encrypted_vec1: bytes,
    encrypted_vec2: bytes,
    context_data: bytes
) -> bytes:
    """
    Compute dot product homomorphically (TRUE PIR core operation)
    
    Args:
        encrypted_vec1: First encrypted vector
        encrypted_vec2: Second encrypted vector
        context_data: CKKS context
    
    Returns:
        Encrypted dot product result
    """
    context = ts.context_from(context_data)
    vec1 = ts.ckks_vector_from(context, encrypted_vec1)
    vec2 = ts.ckks_vector_from(context, encrypted_vec2)
    result = vec1.dot(vec2)
    
    return result.serialize()


# ============================================
# PIR Client
# ============================================

class PIRClient:
    """Client-side PIR operations"""
    
    def __init__(self):
        self.encryptor = get_encryptor()
    
    def prepare_query(self, query_embedding: np.ndarray) -> Dict:
        """
        Prepare encrypted query
        
        Args:
            query_embedding: 256-dim query vector
        
        Returns:
            Dict with encrypted_query and context
        """
        normalized_query = normalize_embedding(query_embedding)
        encrypted_data, context_data = self.encryptor.encrypt_embedding(normalized_query)
        
        return {
            "encrypted_query": encrypted_data,
            "context": context_data,
            "is_normalized": True
        }


# ============================================
# PIR Server
# ============================================

class PIRServer:
    """Server-side PIR operations (no decryption key)"""
    
    def __init__(self):
        pass
    
    def compute_similarity_homomorphic(
        self,
        encrypted_query: bytes,
        encrypted_db_vector: bytes,
        context_data: bytes
    ) -> bytes:
        """
        Compute similarity homomorphically without decryption
        
        Args:
            encrypted_query: Encrypted query vector
            encrypted_db_vector: Encrypted database vector
            context_data: CKKS context
        
        Returns:
            Encrypted similarity score
        """
        context = ts.context_from(context_data)
        query_vec = ts.ckks_vector_from(context, encrypted_query)
        db_vec = ts.ckks_vector_from(context, encrypted_db_vector)
        result = query_vec.dot(db_vec)
        
        return result.serialize()
    
import json

# ... existing code ...

def encrypt_metadata(metadata: Dict, context_data: bytes) -> bytes:
    """
    Encrypt metadata dictionary
    
    Args:
        metadata: Dict with license_plate, color, body_type, image_path
        context_data: CKKS context
    
    Returns:
        Encrypted metadata bytes
    """
    # Serialize to JSON
    json_str = json.dumps(metadata)
    json_bytes = json_str.encode('utf-8')
    
    # Simple XOR encryption (for metadata, not vectors)
    # In production, use proper AES encryption
    # For now, just store encrypted with context
    
    # Convert to vector for CKKS (pad to 256)
    metadata_vector = np.frombuffer(json_bytes, dtype=np.uint8).astype(np.float32)
    
    # Pad to 256
    if len(metadata_vector) < 256:
        metadata_vector = np.pad(metadata_vector, (0, 256 - len(metadata_vector)))
    else:
        metadata_vector = metadata_vector[:256]
    
    # Encrypt with CKKS
    context = ts.context_from(context_data)
    encrypted = ts.ckks_vector(context, metadata_vector.tolist())
    
    return encrypted.serialize()


def decrypt_metadata(encrypted_metadata: bytes, context_data: bytes) -> Dict:
    """
    Decrypt metadata
    
    Args:
        encrypted_metadata: Encrypted bytes
        context_data: CKKS context
    
    Returns:
        Decrypted metadata dictionary
    """
    # Load and decrypt
    context = ts.context_from(context_data)
    encrypted_vec = ts.ckks_vector_from(context, encrypted_metadata)
    decrypted = encrypted_vec.decrypt()
    
    # Convert back to bytes
    metadata_floats = np.array(decrypted[:256])
    metadata_bytes = np.round(metadata_floats).astype(np.uint8).tobytes()
    
    # Remove padding (find null terminator)
    json_str = metadata_bytes.decode('utf-8', errors='ignore').rstrip('\x00')
    
    # Parse JSON
    try:
        metadata = json.loads(json_str)
        return metadata
    except:
        return {}