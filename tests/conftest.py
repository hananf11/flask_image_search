import logging
import os
import shutil

import numpy as np
import pytest
import torch
import torchvision
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from torch import nn

from flask_image_search import ImageSearch

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s Testing: %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
IMAGE = os.path.join(BASE_PATH, "./test.jpg")

# ------------------------------------------------------------------ #
# Load all three backbones once at session start so tests don't pay
# the weight-download cost repeatedly.
# ------------------------------------------------------------------ #

torch.set_num_threads(1)
_device = torch.device("cpu")


def _load_vgg16():
    m = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT)
    m.classifier = m.classifier[:1]
    for p in m.parameters():
        p.grad = None
    m.to(_device)
    m.train(False)
    return m


def _load_vgg19():
    m = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.DEFAULT)
    m.classifier = m.classifier[:1]
    for p in m.parameters():
        p.grad = None
    m.to(_device)
    m.train(False)
    return m


def _load_inception_v3():
    m = torchvision.models.inception_v3(weights=torchvision.models.Inception_V3_Weights.DEFAULT)
    m.fc = nn.Identity()
    m.aux_logits = False
    for p in m.parameters():
        p.grad = None
    m.to(_device)
    m.train(False)
    return m


_vgg16_model = _load_vgg16()
_vgg19_model = _load_vgg19()
_inception_v3_model = _load_inception_v3()


# ------------------------------------------------------------------ #
# ImageSearch subclasses
# ------------------------------------------------------------------ #

class ImageSearchVGG16(ImageSearch):
    def get_model(self):
        self.device = _device
        return _vgg16_model

    def get_feature_size(self):
        return 4096


class ImageSearchVGG19(ImageSearch):
    def get_model(self):
        self.device = _device
        return _vgg19_model

    def get_feature_size(self):
        return 4096


class ImageSearchInceptionV3(ImageSearch):
    def get_model(self):
        self.device = _device
        return _inception_v3_model

    def get_feature_size(self):
        return 2048

    def get_input_size(self):
        return (299, 299)


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #

@pytest.fixture
def app():
    return Flask(__name__)


@pytest.fixture(ids=["test.db"])
def db(app, tmp_path):
    """Copy committed test.db to tmp_path so vector tables never dirty the fixture."""
    src = os.path.join(BASE_PATH, "test.db")
    dst = tmp_path / "test.db"
    shutil.copy(src, str(dst))
    app.config.update({
        "SQLALCHEMY_DATABASE_URI": f"sqlite:///{dst}",
        "SQLALCHEMY_TRACK_MODIFICATIONS": False
    })
    db = SQLAlchemy(app)
    yield db


@pytest.fixture(params=["vgg16", "vgg19", "inception_v3"])
def image_search(app, request):
    app.config.update({"IMAGE_SEARCH_PATH": "image_search/" + request.param})
    request.getfixturevalue("db")

    if request.param == "vgg16":
        image_search = ImageSearchVGG16(app)
    elif request.param == "vgg19":
        image_search = ImageSearchVGG19(app)
    elif request.param == "inception_v3":
        image_search = ImageSearchInceptionV3(app)
    else:
        image_search = ImageSearch(app)

    yield image_search


@pytest.fixture
def Radio(db):
    class Radio(db.Model):
        id = db.Column(db.Integer, primary_key=True)
        name = db.Column(db.Text)
        images = db.relationship("Image")
    return Radio


def _load_fixture_vectors(image_search, Image):
    """Load pre-computed vectors from fixtures/{namespace}.npz into the vector table.

    Falls back to live inference (slow) if the fixture file is missing so that
    tests still work before generate_fixtures.py has been run.
    """
    fixture_path = os.path.join(BASE_PATH, "fixtures", f"{image_search.namespace}.npz")
    if not os.path.exists(fixture_path):
        import warnings
        warnings.warn(
            f"Fixture file {fixture_path} not found; falling back to live inference. "
            "Run `python tests/generate_fixtures.py` to pre-compute vectors.",
            stacklevel=2,
        )
        image_search.index_model(Image, threaded=False)
        return

    data = np.load(fixture_path)
    conn = image_search.db.session.connection()
    for pk, embedding in zip(data["pks"].tolist(), data["embeddings"]):
        image_search.backend.upsert(conn, Image, int(pk), embedding)
    image_search.db.session.flush()


@pytest.fixture
def Image(db, image_search, Radio):
    @image_search.register()
    class Image(db.Model):
        id = db.Column(db.Integer, primary_key=True)
        path = db.Column(db.String, nullable=False)
        radio_id = db.Column(db.Integer, db.ForeignKey(Radio.id), nullable=False)

    _load_fixture_vectors(image_search, Image)
    return Image
