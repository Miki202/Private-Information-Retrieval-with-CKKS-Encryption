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