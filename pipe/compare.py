"""
Compare two vehicle photos by cosine similarity.

Each image is run through the full pipeline:
    photo -> YOLO vehicle crop -> 256-D embedding (cosine encoder)

Embeddings are L2-normalised inside `encode()`, so cosine similarity
is just a dot product.

Usage:
    python compare.py image_a.jpg image_b.jpg
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from pipeline import process_image


# Test threshold from the encoder's test eval — tune to taste
SAME_VEHICLE_THRESHOLD = 0.45


def compare(path_a: str | Path, path_b: str | Path) -> dict:
    """Run pipeline on both images, return cosine similarity + metadata."""
    prof_a = process_image(path_a)
    prof_b = process_image(path_b)

    if prof_a.embedding is None or prof_b.embedding is None:
        return {
            "ok": False,
            "reason": "no vehicle detected in one or both images",
            "a_detected": prof_a.detected,
            "b_detected": prof_b.detected,
        }

    cos = float(np.dot(prof_a.embedding, prof_b.embedding))
    return {
        "ok":            True,
        "cosine":        cos,
        "threshold":     SAME_VEHICLE_THRESHOLD,
        "same_vehicle":  cos >= SAME_VEHICLE_THRESHOLD,
        "a": {
            "path":   prof_a.image_path,
            "color":  prof_a.color["name"]      if prof_a.color  else None,
            "plates": [p["text"] for p in prof_a.plates],
        },
        "b": {
            "path":   prof_b.image_path,
            "color":  prof_b.color["name"]      if prof_b.color  else None,
            "plates": [p["text"] for p in prof_b.plates],
        },
    }


def _pretty_print(result: dict) -> None:
    if not result["ok"]:
        print(f"\nFAILED: {result['reason']}")
        print(f"  a detected: {result['a_detected']}")
        print(f"  b detected: {result['b_detected']}")
        return

    a, b = result["a"], result["b"]
    print(f"\n=== {Path(a['path']).name}  vs  {Path(b['path']).name} ===")
    print(f"  cosine:        {result['cosine']:.4f}   (threshold = {result['threshold']})")
    print(f"  verdict:       {'SAME vehicle' if result['same_vehicle'] else 'DIFFERENT vehicles'}")
    print(f"  a:  colour={a['color']!s:<8}  plates={a['plates']}")
    print(f"  b:  colour={b['color']!s:<8}  plates={b['plates']}")

    # Bonus signals
    if a["plates"] and b["plates"]:
        plates_match = bool(set(a["plates"]) & set(b["plates"]))
        print(f"  plate match:   {plates_match}")
    if a["color"] and b["color"]:
        print(f"  colour match:  {a['color'] == b['color']}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python compare.py <image_a.jpg> <image_b.jpg>")
        sys.exit(1)

    result = compare(sys.argv[1], sys.argv[2])
    _pretty_print(result)
