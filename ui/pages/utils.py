"""
Помощни функции за Streamlit UI
"""
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import sys
import os

# Добавяме parent директорията в path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

class Letterbox:
    """Resize и padding на изображение до квадрат"""
    def __init__(self, size, fill=0):
        self.size = size
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        scale = min(self.size / w, self.size / h)
        nw, nh = int(w * scale), int(h * scale)

        # Resize
        img = img.resize((nw, nh), Image.BILINEAR)
        
        # Padding
        new_img = Image.new("RGB", (self.size, self.size), (self.fill,)*3)
        new_img.paste(img, ((self.size - nw)//2, (self.size - nh)//2))

        return new_img


def load_encoder_model(model_path: str):
    """
    Зарежда encoder модела
    
    Args:
        model_path: Път до .pth файла
    
    Returns:
        (model, device)
    """
    # Опит да импортнем Encoder класа
    # Промени пътя според твоята структура
    try:
        # Опит 1: От notebooks директорията
        from notebooks.image2vec.AE import Encoder
    except:
        try:
            # Опит 2: От локална дефиниция
            from .encoder import Encoder
        except:
            raise ImportError(
                "Не може да се намери Encoder класа. "
                "Моля копирай Encoder класа в ui/encoder.py"
            )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Encoder(latent_dim=256).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model, device


def encode_image(image: Image.Image, model, device) -> np.ndarray:
    """
    Конвертира изображение в 256-мерен embedding
    
    Args:
        image: PIL Image
        model: Encoder модел
        device: torch device
    
    Returns:
        256-мерен numpy array
    """
    # Transform pipeline
    transform = transforms.Compose([
        Letterbox(256),
        transforms.ToTensor()
    ])
    
    # Преобразуване и inference
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        embedding = model(img_tensor).squeeze().cpu().numpy()
    
    return embedding


def format_bytes(num_bytes: int) -> str:
    """Форматира bytes в human-readable формат"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} TB"


def format_number(num: int) -> str:
    """Форматира число с хиляди separator"""
    return f"{num:,}"