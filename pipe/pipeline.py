"""
End-to-end vehicle profiling pipeline.

Input:  a single JPG photo of a vehicle.
Output: a structured profile combining four independent models.

Stages:
    1. Vehicle detection  --  generic COCO-pretrained YOLOv8n (classes car/bus/truck)
                              Crops the highest-confidence vehicle from the photo.
    2. Colour extraction  --  HSV-rules on the vehicle crop  (car_color_example)
    3. Vector embedding   --  cosine encoder on the vehicle crop  (img2vec_example)
    4. Licence-plate OCR  --  custom YOLO + CRNN on the *original* photo
                              (plate detector trained on full images, not crops)
                              (car_plate_example)

Stages 2-4 are independent — each can fail without breaking the others.
The pipeline returns whatever subset succeeded.
"""
from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Make the per-stage helpers importable from sibling `notebooks/` directory.
# ---------------------------------------------------------------------------
ROOT      = Path(__file__).parent.parent.resolve()
NOTEBOOKS = ROOT / "notebooks"
sys.path.insert(0, str(NOTEBOOKS))

from car_color_example import extract_color                # noqa: E402
from car_plate_example  import read_plates                 # noqa: E402
from img2vec_example    import encode                      # noqa: E402


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
YOLO_WEIGHTS      = "yolov8n.pt"          # generic COCO model (downloaded on first run)
VEHICLE_CLASS_IDS = [2, 5, 7]             # car=2, bus=5, truck=7 (COCO indices)
VEHICLE_CONF      = 0.30
VEHICLE_PAD_RATIO = 0.03                  # mild padding to avoid clipping bumpers


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------
@dataclass
class VehicleProfile:
    image_path:    str
    detected:      bool                       = False
    bbox:          Optional[tuple]            = None      # (x1, y1, x2, y2) in pixels
    detection_conf: float                     = 0.0
    color:         Optional[dict]             = None      # extract_color() output
    embedding:     Optional[np.ndarray]       = field(default=None, repr=False)  # (256,)
    plates:        list                       = field(default_factory=list)      # [{"text", "crop"}, ...]

    def to_dict(self) -> dict:
        d = asdict(self)
        # numpy is not json-serialisable; drop the heavy parts for printing
        if self.embedding is not None:
            d["embedding"] = self.embedding.tolist()
        # PIL crops in plates are not serialisable
        d["plates"] = [{"text": p["text"]} for p in self.plates]
        return d


# ---------------------------------------------------------------------------
# Stage 1 — vehicle detection
# ---------------------------------------------------------------------------
_vehicle_detector: Optional[YOLO] = None


def _get_vehicle_detector() -> YOLO:
    global _vehicle_detector
    if _vehicle_detector is None:
        _vehicle_detector = YOLO(YOLO_WEIGHTS)
    return _vehicle_detector


def detect_vehicle(pil: Image.Image) -> Optional[dict]:
    """
    Run COCO-YOLO on the photo and return the highest-confidence vehicle.

    Returns:
        {"bbox": (x1, y1, x2, y2), "conf": float, "crop": PIL.Image} or None.
    """
    detector = _get_vehicle_detector()
    result = detector.predict(
        source=pil,
        conf=VEHICLE_CONF,
        classes=VEHICLE_CLASS_IDS,
        verbose=False,
    )[0]
    if result.boxes is None or len(result.boxes) == 0:
        return None

    boxes = result.boxes.xyxy.cpu().numpy()      # (N, 4)
    confs = result.boxes.conf.cpu().numpy()      # (N,)
    best  = int(np.argmax(confs))
    x1, y1, x2, y2 = boxes[best]

    # Apply padding, clamp to image bounds
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


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------
def process_image(image_path: str | Path) -> VehicleProfile:
    """
    Run the full pipeline on a single image and return a VehicleProfile.

    Failures in any one stage are caught and logged; the rest of the
    profile is still populated.
    """
    image_path = str(image_path)
    profile = VehicleProfile(image_path=image_path)
    pil = Image.open(image_path).convert("RGB")

    # 1) Vehicle detection (gates stages 2 & 3)
    vehicle = detect_vehicle(pil)
    if vehicle is not None:
        profile.detected       = True
        profile.bbox           = vehicle["bbox"]
        profile.detection_conf = vehicle["conf"]
        crop = vehicle["crop"]

        # 2) Colour
        try:
            profile.color = extract_color(crop)
        except Exception as e:
            print(f"[color] failed: {e}")

        # 3) Embedding
        try:
            emb = encode(crop)                  # torch.Tensor (256,)
            profile.embedding = emb.detach().cpu().numpy()
        except Exception as e:
            print(f"[embedding] failed: {e}")
    else:
        print("[vehicle] no car/bus/truck detected")

    # 4) Plate OCR (independent — runs on full image)
    try:
        profile.plates = read_plates(pil)
    except Exception as e:
        print(f"[plate] failed: {e}")

    return profile


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
def _pretty_print(profile: VehicleProfile) -> None:
    print(f"\n=== {profile.image_path} ===")
    if not profile.detected:
        print("  vehicle: NOT DETECTED")
    else:
        x1, y1, x2, y2 = profile.bbox
        print(f"  vehicle:    detected (conf={profile.detection_conf:.2f}, bbox=({x1},{y1},{x2},{y2}))")
    if profile.color is not None:
        c = profile.color
        print(f"  colour:     {c['name']:10s}  hex={c['hex']}  share={c['share']*100:.0f}%")
    if profile.embedding is not None:
        e = profile.embedding
        print(f"  embedding:  shape={e.shape}  norm={np.linalg.norm(e):.4f}")
    if profile.plates:
        for i, p in enumerate(profile.plates):
            print(f"  plate[{i}]:   text={p['text']!r}  size={p['crop'].size}")
    else:
        print("  plate:      none detected")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python pipeline.py <image1.jpg> [<image2.jpg> ...]")
        sys.exit(0)

    for path in sys.argv[1:]:
        profile = process_image(path)
        _pretty_print(profile)
