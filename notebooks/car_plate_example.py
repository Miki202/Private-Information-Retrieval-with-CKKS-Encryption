"""
License-plate OCR — minimal usage example.

End-to-end pipeline:
    car photo  -->  YOLO plate detector  -->  cropped plate  -->  CRNN  -->  text

Two models are involved:
  - YOLO weights:  notebooks/detectors/plates/car-plate-best.pt
  - CRNN weights:  notebooks/plate/crnn_epoch_60.pth

The CRNN architecture and alphabet are imported from `car_plate.py`
(same file used during training).
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

from car_plate import (
    BLANK_IDX,
    CRNN,
    NUM_CLASSES,
    idx_to_char,
)


HERE        = Path(__file__).parent.resolve()       # notebooks/
YOLO_PATH   = HERE / "detectors" / "plates" / "car-plate-best.pt"
CRNN_PATH   = HERE / "plate" / "crnn_epoch_60.pth"

IMG_SIZE        = 128                      # CRNN was trained on 128x128 RGB
PLATE_CONF      = 0.30                     # YOLO confidence for plate detection
PLATE_PAD_RATIO = 0.05                     # mild padding around the bbox
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"


plate_preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def load_detector(weights_path: Path = YOLO_PATH) -> YOLO:
    """YOLOv8 fine-tuned for license plate detection."""
    return YOLO(str(weights_path))


def load_crnn(weights_path: Path = CRNN_PATH) -> CRNN:
    """CRNN OCR model (CNN + BiLSTM + CTC)."""
    model = CRNN(img_height=IMG_SIZE, num_channels=3, hidden_size=256).to(DEVICE)
    state = torch.load(weights_path, map_location=DEVICE)
    # state may be a raw state_dict or a {"model": ...} checkpoint
    if isinstance(state, dict) and "model" in state and not any(k.startswith("cnn") for k in state):
        state = state["model"]
    model.load_state_dict(state)
    model.eval()
    return model


# Lazy global instances (loaded on first use)
_detector: YOLO | None = None
_crnn:     CRNN | None = None


def _get_detector() -> YOLO:
    global _detector
    if _detector is None:
        _detector = load_detector()
    return _detector


def _get_crnn() -> CRNN:
    global _crnn
    if _crnn is None:
        _crnn = load_crnn()
    return _crnn


# ---------------------------------------------------------------------------
# CTC greedy decode
# ---------------------------------------------------------------------------
def ctc_greedy_decode(logits: torch.Tensor) -> list[str]:
    """
    Args:
        logits: (T, B, NUM_CLASSES) raw scores from CRNN.forward
    Returns:
        list of B decoded strings.
    """
    # (T, B) — best class per timestep
    best = logits.argmax(dim=2).detach().cpu().numpy()      # (T, B)
    T, B = best.shape

    decoded: list[str] = []
    for b in range(B):
        seq = best[:, b]
        # Collapse repeats then drop blanks (standard CTC greedy decode)
        out = []
        prev = -1
        for idx in seq:
            if idx != prev and idx != BLANK_IDX:
                out.append(idx_to_char.get(int(idx), ""))
            prev = idx
        decoded.append("".join(out))
    return decoded


# ---------------------------------------------------------------------------
# Plate detection + cropping
# ---------------------------------------------------------------------------
def detect_plates(
    pil_image: Image.Image,
    conf: float = PLATE_CONF,
    pad_ratio: float = PLATE_PAD_RATIO,
) -> list[Image.Image]:
    """
    Run YOLO and return cropped plate regions, sorted by descending confidence.

    Returns an empty list when no plate is detected.
    """
    detector = _get_detector()
    result = detector.predict(source=pil_image, conf=conf, verbose=False)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return []

    W, H = pil_image.size
    boxes = result.boxes.xyxy.cpu().numpy()                 # (N, 4)
    confs = result.boxes.conf.cpu().numpy()                 # (N,)
    order = np.argsort(-confs)                              # high confidence first

    crops: list[Image.Image] = []
    for i in order:
        x1, y1, x2, y2 = boxes[i]
        bw, bh = x2 - x1, y2 - y1
        px, py = bw * pad_ratio, bh * pad_ratio
        nx1 = max(0, int(round(x1 - px)))
        ny1 = max(0, int(round(y1 - py)))
        nx2 = min(W, int(round(x2 + px)))
        ny2 = min(H, int(round(y2 + py)))
        crops.append(pil_image.crop((nx1, ny1, nx2, ny2)))
    return crops


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------
@torch.no_grad()
def ocr_plate(plate_crop: Image.Image) -> str:
    """Run CRNN on an already-cropped plate image; returns decoded text."""
    crnn = _get_crnn()
    x = plate_preprocess(plate_crop.convert("RGB")).unsqueeze(0).to(DEVICE)
    logits = crnn(x)                                        # (T, 1, NUM_CLASSES)
    return ctc_greedy_decode(logits)[0]


@torch.no_grad()
def ocr_plates_batch(plate_crops: list[Image.Image]) -> list[str]:
    """Batched OCR for many plate crops at once."""
    if not plate_crops:
        return []
    crnn = _get_crnn()
    batch = torch.stack([plate_preprocess(c.convert("RGB")) for c in plate_crops]).to(DEVICE)
    logits = crnn(batch)                                    # (T, B, NUM_CLASSES)
    return ctc_greedy_decode(logits)


# ---------------------------------------------------------------------------
# End-to-end: image -> list of plate strings
# ---------------------------------------------------------------------------
def read_plates(image_path: str | Path | Image.Image) -> list[dict]:
    """
    Detect every plate in `image_path` and OCR each one.

    Returns a list of dicts:
        [{"text": "AA1234BB", "crop": <PIL.Image>}, ...]
    sorted by detection confidence (most confident first).
    """
    pil = image_path if isinstance(image_path, Image.Image) else Image.open(image_path).convert("RGB")
    crops = detect_plates(pil)
    if not crops:
        return []
    texts = ocr_plates_batch(crops)
    return [{"text": t, "crop": c} for t, c in zip(texts, crops)]


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("usage: python example.py <image1> [<image2> ...]")
        sys.exit(0)

    for path in sys.argv[1:]:
        print(f"\n=== {path} ===")
        results = read_plates(path)
        if not results:
            print("  no plates detected")
            continue
        for i, r in enumerate(results):
            print(f"  [{i}] text={r['text']!r}  size={r['crop'].size}")
