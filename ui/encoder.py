import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import torch.nn as nn

class Letterbox:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        w, h = img.size
        scale = min(self.size / w, self.size / h)
        nw, nh = int(w * scale), int(h * scale)

        img = img.resize((nw, nh), Image.BILINEAR)

        new_img = Image.new("RGB", (self.size, self.size))
        new_img.paste(img, ((self.size - nw)//2, (self.size - nh)//2))
        return new_img


class Encoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
        )
        self.fc = nn.Linear(128 * 32 * 32, latent_dim)

    def forward(self, x):
        x = self.net(x)
        x = x.flatten(1)
        return self.fc(x)


def load_model(path="encoder.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Encoder().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model, device


def encode_image(image, model, device):
    transform = transforms.Compose([
        Letterbox(256),
        transforms.ToTensor()
    ])

    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model(x).squeeze().cpu().numpy()

    return emb