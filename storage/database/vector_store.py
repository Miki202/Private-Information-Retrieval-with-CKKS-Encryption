"""
FAISS vector store за plain и encrypted embeddings
"""
import faiss
import numpy as np
import os
import json
from typing import List, Tuple, Optional, Dict

# Data директория за FAISS indexes
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

class FAISSVectorStore:
    """
    FAISS index wrapper с mapping към PostgreSQL vehicle IDs
    """
    
    def __init__(self, mode: str = "plain", dimension: int = 256):
        """
        Args:
            mode: "plain" или "encrypted"
            dimension: Vector размер (256)
        """
        self.mode = mode
        self.dimension = dimension
        
        # Файлове
        self.index_path = os.path.join(DATA_DIR, f"{mode}_index.faiss")
        self.mapping_path = os.path.join(DATA_DIR, f"{mode}_mapping.json")
        
        # FAISS index
        self.index = None
        
        # Mapping: {faiss_id: vehicle_id (PostgreSQL)}
        self.id_mapping = {}
        
        # Load ако съществува
        self.load()
    
    def _create_index(self):
        """Създава нов FAISS HNSW index"""
        self.index = faiss.IndexHNSWFlat(self.dimension, 32)
        self.index.hnsw.efConstruction = 40
        self.index.hnsw.efSearch = 16
        print(f"✓ Нов {self.mode} FAISS index създаден")
    
    def add_vector(self, vehicle_id: int, embedding: np.ndarray) -> int:
        """
        Добавя vector в FAISS
        
        Args:
            vehicle_id: PostgreSQL vehicle ID
            embedding: 256-dim vector
        
        Returns:
            faiss_id: ID в FAISS index
        """
        if self.index is None:
            self._create_index()
        
        # Format
        if isinstance(embedding, list):
            embedding = np.array(embedding)
        
        if len(embedding.shape) == 1:
            embedding = embedding.reshape(1, -1)
        
        embedding = embedding.astype(np.float32)
        
        # FAISS ID = текущ брой vectors
        faiss_id = self.index.ntotal
        
        # Добави в FAISS
        self.index.add(embedding)
        
        # Mapping
        self.id_mapping[faiss_id] = vehicle_id
        
        print(f"  FAISS: vector {faiss_id} → vehicle {vehicle_id}")
        
        return faiss_id
    
    def search(self, query: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Търси k nearest neighbors
        
        Returns:
            (distances, vehicle_ids) - vehicle_ids са от PostgreSQL
        """
        if self.index is None or self.index.ntotal == 0:
            return np.array([]), np.array([])
        
        # Format query
        if isinstance(query, list):
            query = np.array(query)
        if len(query.shape) == 1:
            query = query.reshape(1, -1)
        query = query.astype(np.float32)
        
        # FAISS search
        distances, faiss_ids = self.index.search(query, min(k, self.index.ntotal))
        
        # Map FAISS IDs → PostgreSQL vehicle IDs
        vehicle_ids = []
        for fid in faiss_ids[0]:
            if fid != -1:
                vehicle_ids.append(self.id_mapping.get(int(fid), -1))
            else:
                vehicle_ids.append(-1)
        
        return distances[0], np.array(vehicle_ids)
    
    def remove_vector(self, faiss_id: int):
        """
        Премахва vector от mapping
        FAISS не поддържа deletion - трябва rebuild
        """
        if faiss_id in self.id_mapping:
            del self.id_mapping[faiss_id]
    
    def rebuild_from_db(self, vehicles_data: List[Tuple[int, np.ndarray]]):
        """
        Rebuild index от scratch
        
        Args:
            vehicles_data: List of (vehicle_id, embedding) tuples
        """
        print(f"Rebuilding {self.mode} FAISS index...")
        self._create_index()
        self.id_mapping = {}
        
        for vehicle_id, embedding in vehicles_data:
            self.add_vector(vehicle_id, embedding)
        
        self.save()
        print(f"✓ Index rebuilt: {self.index.ntotal} vectors")
    
    def save(self):
        """Съхранява index и mappings"""
        if self.index is not None and self.index.ntotal > 0:
            faiss.write_index(self.index, self.index_path)
            print(f"✓ {self.mode} index saved: {self.index_path}")
        
        mapping_str = {str(k): v for k, v in self.id_mapping.items()}
        with open(self.mapping_path, 'w') as f:
            json.dump(mapping_str, f)
        print(f"✓ {self.mode} mapping saved")
    
    def load(self):
        """Зарежда index и mappings"""
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            print(f"✓ {self.mode} FAISS loaded: {self.index.ntotal} vectors")
        
        if os.path.exists(self.mapping_path):
            with open(self.mapping_path, 'r') as f:
                mapping_str = json.load(f)
            self.id_mapping = {int(k): v for k, v in mapping_str.items()}
            print(f"✓ {self.mode} mapping loaded: {len(self.id_mapping)} entries")
    
    def get_stats(self) -> Dict:
        """Статистика"""
        if self.index is None:
            return {"total_vectors": 0}
        return {
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "mapping_entries": len(self.id_mapping)
        }


# Глобални инстанции
_plain_store = None
_encrypted_store = None

def get_plain_store() -> FAISSVectorStore:
    global _plain_store
    if _plain_store is None:
        _plain_store = FAISSVectorStore(mode="plain")
    return _plain_store

def get_encrypted_store() -> FAISSVectorStore:
    global _encrypted_store
    if _encrypted_store is None:
        _encrypted_store = FAISSVectorStore(mode="encrypted")
    return _encrypted_store

def save_all_stores():
    """Съхранява всички stores"""
    if _plain_store:
        _plain_store.save()
    if _encrypted_store:
        _encrypted_store.save()