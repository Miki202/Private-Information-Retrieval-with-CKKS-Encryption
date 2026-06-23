import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

class Letterbox:
    """Променя размера на изображението, запазвайки съотношението чрез запълване (padding)"""
    def __init__(self, size, fill=0):
        self.size = size
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        scale = min(self.size / w, self.size / h)
        nw, nh = int(w * scale), int(h * scale)

        img = img.resize((nw, nh), Image.Resampling.BILINEAR)
        
        new_img = Image.new("RGB", (self.size, self.size), (self.fill,)*3)
        new_img.paste(img, ((self.size - nw)//2, (self.size - nh)//2))
        return new_img

def load_encoder_model(model_path: str):
    try:
        from notebooks.image2vec.AE import Encoder
    except ImportError:
        try:
            from .encoder import Encoder
        except ImportError:
            raise ImportError("Класът Encoder не може да бъде намерен. Уверете се, че съществува в ui/encoder.py")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Encoder(latent_dim=256).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

def encode_image(image: Image.Image, model, device) -> np.ndarray:
    transform = transforms.Compose([
        Letterbox(256),
        transforms.ToTensor()
    ])
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(img_tensor).squeeze().cpu().numpy()
    return embedding