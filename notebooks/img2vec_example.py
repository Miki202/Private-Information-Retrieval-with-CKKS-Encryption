from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoModel


HF_REPO_ID = "quebeccyb/vehitv-cropped" 
IMG_SIZE   = 256
THRESHOLD  = 0.45  
DEVICE= "cuda" if torch.cuda.is_available() else "cpu"


class Letterbox:
    def __init__(self, size: int, fill: int = 0):
        self.size = size
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        scale = min(self.size / w, self.size / h)
        nw, nh = int(w * scale), int(h * scale)
        img = img.resize((nw, nh), Image.BILINEAR)
        canvas = Image.new("RGB", (self.size, self.size), (self.fill,) * 3)
        canvas.paste(img, ((self.size - nw) // 2, (self.size - nh) // 2))
        return canvas


preprocess = transforms.Compose([
    Letterbox(IMG_SIZE),
    transforms.ToTensor(),])
model = AutoModel.from_pretrained(HF_REPO_ID, trust_remote_code=True).to(DEVICE).eval()

@torch.no_grad()
def encode(pil_image: Image.Image) -> torch.Tensor:
    x = preprocess(pil_image.convert("RGB")).unsqueeze(0).to(DEVICE)
    emb = model(x).squeeze(0)
    return F.normalize(emb, dim=0)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a * b).sum())

def compare(path_a: str | Path, path_b: str | Path) -> dict:
    ea = encode(Image.open(path_a))
    eb = encode(Image.open(path_b))
    sim = cosine(ea, eb)
    return {
        "cosine": sim,
        "same_vehicle": sim >= THRESHOLD,
        "threshold": THRESHOLD,
    }

class _ImageListDataset(Dataset):
    def __init__(self, paths: list[Path], tf):
        self.paths = paths
        self.tf = tf

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.tf(Image.open(self.paths[idx]).convert("RGB"))


@torch.no_grad()
def encode_many(paths: list[Path], batch_size: int = 64, num_workers: int = 2) -> np.ndarray:
    ds = _ImageListDataset(paths, preprocess)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers)
    chunks = []
    for batch in tqdm(loader, desc="encoding"):
        e = model(batch.to(DEVICE))
        e = F.normalize(e, dim=1)
        chunks.append(e.cpu().numpy().astype("float32"))
    return np.concatenate(chunks, axis=0)

def search(query_img: Image.Image, gallery_emb: np.ndarray, k: int = 5) -> list[tuple[int, float]]:
    q = encode(query_img).cpu().numpy().astype("float32")
    sims = gallery_emb @ q                       # (N,)
    top = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i])) for i in top]

if __name__ == "__main__":
    gallery_dir = Path("gallery")
    if gallery_dir.exists():
        gallery_paths = sorted(gallery_dir.glob("*.jpg"))
        if gallery_paths:
            gallery_emb = encode_many(gallery_paths)
            np.save("gallery_emb.npy", gallery_emb)
            print(f"Indexed {len(gallery_paths)} images -> gallery_emb.npy")

            query = Image.open(gallery_paths[0])
            print("\nTop-5 nearest to first gallery image:")
            for idx, score in search(query, gallery_emb, k=5):
                print(f"  {score:.4f}  {gallery_paths[idx].name}")
    else:
        print(f"(create a '{gallery_dir}/' folder with .jpg images to run the gallery demo)")
