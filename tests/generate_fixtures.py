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
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from keras.applications import inception_v3, vgg16, vgg19
from keras.models import Model as KerasModel

from flask_image_search import ImageSearch

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
FIXTURES_PATH = os.path.join(BASE_PATH, "fixtures")
os.makedirs(FIXTURES_PATH, exist_ok=True)


def _make_is_class(keras_model, preprocess_fn):
    class _IS(ImageSearch):
        @staticmethod
        def create_keras_model():
            return keras_model

        @staticmethod
        def preprocess_image_array(arr):
            return preprocess_fn(arr)

    return _IS


def _build_app(namespace):
    app = Flask(__name__)
    app.config.update(
        SQLALCHEMY_DATABASE_URI=f"sqlite:///{BASE_PATH}/test.db",
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        IMAGE_SEARCH_NAMESPACE=namespace,  # dummy — we don't use the backend here
    )
    SQLAlchemy(app)
    return app


def generate(name, keras_model, preprocess_fn):
    print(f"[{name}] loading model...", flush=True)
    ISClass = _make_is_class(keras_model, preprocess_fn)

    app = Flask(__name__)
    app.config.update(
        SQLALCHEMY_DATABASE_URI=f"sqlite:///{BASE_PATH}/test.db",
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
    )
    db = SQLAlchemy(app)
    is_instance = ISClass(app)
    namespace = is_instance.namespace

    out_path = os.path.join(FIXTURES_PATH, f"{namespace}.npz")
    if os.path.exists(out_path):
        print(f"[{name}] {out_path} already exists, skipping (delete to regenerate)")
        return namespace

    # Read image rows directly — no need to involve SQLAlchemy ORM here
    conn = sqlite3.connect(os.path.join(BASE_PATH, "test.db"))
    rows = conn.execute("SELECT id, path FROM image ORDER BY id").fetchall()
    conn.close()

    print(f"[{name}] extracting features for {len(rows)} images...", flush=True)
    pks = []
    embeddings = []
    for i, (pk, path) in enumerate(rows):
        from PIL import Image as PILImage
        img = PILImage.open(os.path.join(BASE_PATH, path))
        feat = is_instance.feature_extract(img)
        pks.append(pk)
        embeddings.append(feat)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(rows)}", flush=True)

    np.savez(out_path, pks=np.array(pks, dtype=np.int64), embeddings=np.array(embeddings, dtype=np.float32))
    print(f"[{name}] saved {len(pks)} vectors → fixtures/{namespace}.npz")
    return namespace


if __name__ == "__main__":
    _vgg16 = vgg16.VGG16(weights="imagenet")
    _vgg16 = KerasModel(_vgg16.input, _vgg16.get_layer("fc1").output)

    _vgg19 = vgg19.VGG19(weights="imagenet")
    _vgg19 = KerasModel(_vgg19.input, _vgg19.get_layer("fc1").output)

    _inception = inception_v3.InceptionV3(weights="imagenet")
    _inception = KerasModel(_inception.input, _inception.get_layer("avg_pool").output)

    generate("vgg16", _vgg16, vgg16.preprocess_input)
    generate("vgg19", _vgg19, vgg19.preprocess_input)
    generate("inception_v3", _inception, inception_v3.preprocess_input)
