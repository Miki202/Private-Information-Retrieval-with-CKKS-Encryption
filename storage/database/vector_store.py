"""
FAISS vector store for PLAIN mode only
TRUE PIR mode does NOT use FAISS (uses PostgreSQL with homomorphic operations)
"""
import faiss
import numpy as np
import os
import json
from typing import List, Tuple, Dict

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

class FAISSVectorStore:
    """FAISS index wrapper for plain (unencrypted) embeddings"""
    
    def __init__(self, dimension: int = 256):
        """
        Args:
            dimension: Vector dimension (default 256)
        """
        self.dimension = dimension
        
        self.index_path = os.path.join(DATA_DIR, "plain_index.faiss")
        self.mapping_path = os.path.join(DATA_DIR, "plain_mapping.json")
        
        self.index = None
        self.id_mapping = {}
        
        self.load()
    
    def _create_index(self):
        """Create new FAISS HNSW index"""
        self.index = faiss.IndexHNSWFlat(self.dimension, 32)
        self.index.hnsw.efConstruction = 40
        self.index.hnsw.efSearch = 16
    
    def add_vector(self, vehicle_id: int, embedding: np.ndarray) -> int:
        """
        Add vector to FAISS index
        
        Args:
            vehicle_id: PostgreSQL vehicle ID
            embedding: 256-dim vector
        
        Returns:
            faiss_id: ID in FAISS index
        """
        if self.index is None:
            self._create_index()
        
        if isinstance(embedding, list):
            embedding = np.array(embedding)
        
        if len(embedding.shape) == 1:
            embedding = embedding.reshape(1, -1)
        
        embedding = embedding.astype(np.float32)
        
        faiss_id = self.index.ntotal
        self.index.add(embedding)
        self.id_mapping[faiss_id] = vehicle_id
        
        return faiss_id
    
    def search(self, query: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search k nearest neighbors
        
        Args:
            query: Query vector
            k: Number of results
        
        Returns:
            (distances, vehicle_ids) - vehicle_ids are PostgreSQL IDs
        """
        if self.index is None or self.index.ntotal == 0:
            return np.array([]), np.array([])
        
        if isinstance(query, list):
            query = np.array(query)
        if len(query.shape) == 1:
            query = query.reshape(1, -1)
        query = query.astype(np.float32)
        
        distances, faiss_ids = self.index.search(query, min(k, self.index.ntotal))
        
        vehicle_ids = []
        for fid in faiss_ids[0]:
            if fid != -1:
                vehicle_ids.append(self.id_mapping.get(int(fid), -1))
            else:
                vehicle_ids.append(-1)
        
        return distances[0], np.array(vehicle_ids)
    
    def remove_vector(self, faiss_id: int):
        """Remove vector from mapping (FAISS doesn't support deletion)"""
        if faiss_id in self.id_mapping:
            del self.id_mapping[faiss_id]
    
    def rebuild_from_db(self, vehicles_data: List[Tuple[int, np.ndarray]]):
        """
        Rebuild index from scratch
        
        Args:
            vehicles_data: List of (vehicle_id, embedding) tuples
        """
        self._create_index()
        self.id_mapping = {}
        
        for vehicle_id, embedding in vehicles_data:
            self.add_vector(vehicle_id, embedding)
        
        self.save()
    
    def save(self):
        """Save index and mappings to disk"""
        if self.index is not None and self.index.ntotal > 0:
            faiss.write_index(self.index, self.index_path)
        
        mapping_str = {str(k): v for k, v in self.id_mapping.items()}
        with open(self.mapping_path, 'w') as f:
            json.dump(mapping_str, f)
    
    def load(self):
        """Load index and mappings from disk"""
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        
        if os.path.exists(self.mapping_path):
            with open(self.mapping_path, 'r') as f:
                mapping_str = json.load(f)
            self.id_mapping = {int(k): v for k, v in mapping_str.items()}
    
    def get_stats(self) -> Dict:
        """Get index statistics"""
        if self.index is None:
            return {
                "total_vectors": 0,
                "dimension": self.dimension,
                "mapping_entries": 0
            }
        
        return {
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "mapping_entries": len(self.id_mapping)
        }


# Global singleton for plain store only
_plain_store = None

def get_plain_store() -> FAISSVectorStore:
    """Get or create plain FAISS store (used for plain mode search only)"""
    global _plain_store
    if _plain_store is None:
        _plain_store = FAISSVectorStore()
    return _plain_store

def save_all_stores():
    """Save FAISS store to disk"""
    if _plain_store:
        _plain_store.save()