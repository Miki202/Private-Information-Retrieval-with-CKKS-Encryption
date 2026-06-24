import sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from notebooks.img2vec_example import encode

def encode_image(image: Image.Image) -> np.ndarray:
    """
    Wrapper за encode() от notebooks/img2vec_example.py
    
    Args:
        image: PIL.Image (RGB)
        model: Игнорира се (за обратна съвместимost)
        device: Игнорира се (за обратна съвместимost)
    
    Returns:
        numpy.ndarray (256,) dtype=float32, L2-нормализиран
    """
    embedding_tensor = encode(image)
    return embedding_tensor.cpu().numpy()