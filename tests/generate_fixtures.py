"""Pre-compute feature vectors for test fixtures.

Run this once (or whenever test.db images change) to generate the cached
feature vectors used by the test suite, eliminating CNN inference on every run.

Usage:
    python tests/generate_fixtures.py
"""
# ruff: noqa: E402

import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torchvision
from flask_image_search.helper import extract_features
from PIL import Image as PILImage
from torch import nn

BASE_PATH = Path(__file__).resolve().parent
FIXTURES_PATH = BASE_PATH / "fixtures"
FIXTURES_PATH.mkdir(parents=True, exist_ok=True)

torch.set_num_threads(torch.get_num_threads())  # use all cores during generation
DEVICE = torch.device("cpu")


def generate(name, namespace, model, preprocess):
    out_path = FIXTURES_PATH / f"{namespace}.npz"
    if out_path.exists():
        print(f"[{name}] {out_path.name} already exists, skipping (delete to regenerate)")
        return

    rows = sqlite3.connect(BASE_PATH / "test.db").execute(
        "SELECT id, path FROM image ORDER BY id"
    ).fetchall()

    print(f"[{name}] extracting features for {len(rows)} images...", flush=True)
    pks, embeddings = [], []
    for i, (pk, path) in enumerate(rows):
        img = PILImage.open(BASE_PATH / path)
        feat = extract_features(model, preprocess, img, DEVICE)
        pks.append(pk)
        embeddings.append(feat)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(rows)}", flush=True)

    np.savez(
        out_path,
        pks=np.array(pks, dtype=np.int64),
        embeddings=np.array(embeddings, dtype=np.float32),
    )
    print(f"[{name}] saved {len(pks)} vectors -> fixtures/{namespace}.npz")


if __name__ == "__main__":
    # VGG16
    m = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT)
    m.classifier = m.classifier[:1]
    m.train(False)
    generate("vgg16", "vgg16-fc1", m, torchvision.models.VGG16_Weights.DEFAULT.transforms())

    # VGG19
    m = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.DEFAULT)
    m.classifier = m.classifier[:1]
    m.train(False)
    generate("vgg19", "vgg19-fc1", m, torchvision.models.VGG19_Weights.DEFAULT.transforms())

    # InceptionV3
    m = torchvision.models.inception_v3(weights=torchvision.models.Inception_V3_Weights.DEFAULT)
    m.fc = nn.Identity()
    m.aux_logits = False
    m.train(False)
    generate("inception_v3", "inception_v3-pool", m, torchvision.models.Inception_V3_Weights.DEFAULT.transforms())
