from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
ROOT      = Path(__file__).parent.parent.resolve()
NOTEBOOKS = ROOT / "notebooks"
sys.path.insert(0, str(NOTEBOOKS))

from car_color_example import extract_color                
from car_plate_example  import read_plates               
from img2vec_example    import encode                     

YOLO_WEIGHTS      = "yolov8n.pt"         
VEHICLE_CLASS_IDS = [2, 5, 7]             
VEHICLE_CONF      = 0.30
VEHICLE_PAD_RATIO = 0.03                  

@dataclass
class VehicleProfile:
    image_path: str
    detected: bool  = False
    bbox:  Optional[tuple]= None      
    detection_conf: float   = 0.0
    color:  Optional[dict]  = None     
    embedding: Optional[np.ndarray]  = field(default=None, repr=False)  
    plates:list = field(default_factory=list)      

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.embedding is not None:
            d["embedding"] = self.embedding.tolist()
        d["plates"] = [{"text": p["text"]} for p in self.plates]
        return d

_vehicle_detector: Optional[YOLO] = None

def _get_vehicle_detector() -> YOLO:
    global _vehicle_detector
    if _vehicle_detector is None:
        _vehicle_detector = YOLO(YOLO_WEIGHTS)
    return _vehicle_detector


def detect_vehicle(pil: Image.Image) -> Optional[dict]:
    detector = _get_vehicle_detector()
    result = detector.predict(
        source=pil,
        conf=VEHICLE_CONF,
        classes=VEHICLE_CLASS_IDS,
        verbose=False,
    )[0]
    if result.boxes is None or len(result.boxes) == 0:
        return None
    boxes = result.boxes.xyxy.cpu().numpy()      
    confs = result.boxes.conf.cpu().numpy()     
    best  = int(np.argmax(confs))
    x1, y1, x2, y2 = boxes[best]

    W, H = pil.size
    bw, bh = x2 - x1, y2 - y1
    px, py = bw * VEHICLE_PAD_RATIO, bh * VEHICLE_PAD_RATIO
    nx1 = max(0, int(round(x1 - px)))
    ny1 = max(0, int(round(y1 - py)))
    nx2 = min(W, int(round(x2 + px)))
    ny2 = min(H, int(round(y2 + py)))
    crop = pil.crop((nx1, ny1, nx2, ny2))

    return {
        "bbox": (nx1, ny1, nx2, ny2),
        "conf": float(confs[best]),
        "crop": crop,
    }

def process_image(image_path: str | Path) -> VehicleProfile:
    image_path = str(image_path)
    profile = VehicleProfile(image_path=image_path)
    pil = Image.open(image_path).convert("RGB")
    vehicle = detect_vehicle(pil)
    if vehicle is not None:
        profile.detected   = True
        profile.bbox   = vehicle["bbox"]
        profile.detection_conf = vehicle["conf"]
        crop = vehicle["crop"]
        try:
            profile.color = extract_color(crop)
        except Exception as e:
            print(f"[color] failed: {e}")
        try:
            emb = encode(crop)              
            profile.embedding = emb.detach().cpu().numpy()
        except Exception as e:
            print(f"[embedding] failed: {e}")
    else:
        print("[vehicle] no car/bus/truck detected")
    try:
        profile.plates = read_plates(pil)
    except Exception as e:
        print(f"[plate] failed: {e}")

    return profile

def _pretty_print(profile: VehicleProfile) -> None:
    print(f"\n=== {profile.image_path} ===")
    if not profile.detected:
        print("  vehicle: NOT DETECTED")
    else:
        x1, y1, x2, y2 = profile.bbox
        print(f"  vehicle: detected (conf={profile.detection_conf:.2f}, bbox=({x1},{y1},{x2},{y2}))")
    if profile.color is not None:
        c = profile.color
        print(f"  colour:  {c['name']:10s}  hex={c['hex']}  share={c['share']*100:.0f}%")
    if profile.embedding is not None:
        e = profile.embedding
        print(f"  embedding:  shape={e.shape}  norm={np.linalg.norm(e):.4f}")
    if profile.plates:
        for i, p in enumerate(profile.plates):
            print(f"  plate[{i}]:text={p['text']!r} size={p['crop'].size}")
    else:
        print("plate: none detected")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python pipeline.py <image1.jpg> [<image2.jpg> ...]")
        sys.exit(0)

    for path in sys.argv[1:]:
        profile = process_image(path)
        _pretty_print(profile)
