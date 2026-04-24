"""Pre-compute feature vectors for test fixtures.

Run this once (or whenever test.db images change) to generate the cached
feature vectors used by the test suite, eliminating CNN inference on every run.

Usage:
    python tests/generate_fixtures.py
"""
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import numpy as np
import torch
import torchvision
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from PIL import Image as PILImage
from torch import nn

from flask_image_search import ImageSearch

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
FIXTURES_PATH = os.path.join(BASE_PATH, "fixtures")
os.makedirs(FIXTURES_PATH, exist_ok=True)

torch.set_num_threads(torch.get_num_threads())  # use all cores during generation


def _make_is_class(model, device, input_size=(224, 224), feature_size=4096):
    class _IS(ImageSearch):
        def get_model(self):
            self.device = device
            return model

        def get_feature_size(self):
            return feature_size

        def get_input_size(self):
            return input_size

    return _IS


def generate(name, is_class):
    app = Flask(__name__)
    app.config.update(
        SQLALCHEMY_DATABASE_URI=f"sqlite:///{BASE_PATH}/test.db",
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
    )
    SQLAlchemy(app)
    is_instance = is_class(app)
    namespace = is_instance.namespace

    out_path = os.path.join(FIXTURES_PATH, f"{namespace}.npz")
    if os.path.exists(out_path):
        print(f"[{name}] {os.path.basename(out_path)} already exists, skipping (delete to regenerate)")
        return

    rows = sqlite3.connect(os.path.join(BASE_PATH, "test.db")).execute(
        "SELECT id, path FROM image ORDER BY id"
    ).fetchall()

    print(f"[{name}] extracting features for {len(rows)} images...", flush=True)
    pks, embeddings = [], []
    for i, (pk, path) in enumerate(rows):
        img = PILImage.open(os.path.join(BASE_PATH, path))
        feat = is_instance.feature_extract(img)
        pks.append(pk)
        embeddings.append(feat)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(rows)}", flush=True)

    np.savez(
        out_path,
        pks=np.array(pks, dtype=np.int64),
        embeddings=np.array(embeddings, dtype=np.float32),
    )
    print(f"[{name}] saved {len(pks)} vectors → fixtures/{namespace}.npz")


if __name__ == "__main__":
    device = torch.device("cpu")

    # VGG16
    m = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT)
    m.classifier = m.classifier[:1]
    m.train(False)
    generate("vgg16", _make_is_class(m, device, (224, 224), 4096))

    # VGG19
    m = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.DEFAULT)
    m.classifier = m.classifier[:1]
    m.train(False)
    generate("vgg19", _make_is_class(m, device, (224, 224), 4096))

    # InceptionV3
    m = torchvision.models.inception_v3(weights=torchvision.models.Inception_V3_Weights.DEFAULT)
    m.fc = nn.Identity()
    m.aux_logits = False
    m.train(False)
    generate("inception_v3", _make_is_class(m, device, (299, 299), 2048))
