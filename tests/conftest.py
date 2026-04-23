import logging
import os
import shutil

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # noqa

import numpy as np
import pytest
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from keras.applications import inception_v3, vgg19, vgg16
from keras.models import Model as KerasModel

from flask_image_search import ImageSearch

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s Testing: %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
IMAGE = os.path.join(BASE_PATH, "./test.jpg")


@pytest.fixture
def app():
    """Fixture that returns an instance of FLask."""
    return Flask(__name__)


@pytest.fixture(ids=["test.db"])
def db(app, tmp_path):
    """Fixture for sqlalchemy.

    Copies the committed test.db to a tmp path so tests can write vector
    tables without dirtying the tracked fixture file.
    """
    src = os.path.join(BASE_PATH, "test.db")
    dst = tmp_path / "test.db"
    shutil.copy(src, str(dst))
    app.config.update({
        "SQLALCHEMY_DATABASE_URI": f"sqlite:///{dst}",
        "SQLALCHEMY_TRACK_MODIFICATIONS": False
    })
    db = SQLAlchemy(app)
    yield db


vgg16_model = vgg16.VGG16(weights="imagenet")
vgg16_model = KerasModel(vgg16_model.input, vgg16_model.get_layer("fc1").output)


class ImageSearchVGG16(ImageSearch):
    """ImageSearch model using vgg16 for feature extraction."""

    @staticmethod
    def create_keras_model():
        """Get instance of the keras model being used."""
        return vgg16_model

    @staticmethod
    def preprocess_image_array(image_array):
        """Preprocess image array."""
        return vgg16.preprocess_input(image_array)


vgg19_model = vgg19.VGG19(weights="imagenet")
vgg19_model = KerasModel(vgg19_model.input, vgg19_model.get_layer("fc1").output)


class ImageSearchVGG19(ImageSearch):
    """ImageSearch model using vgg19 for feature extraction."""

    @staticmethod
    def create_keras_model():
        """Get instance of the keras model being used."""
        return vgg19_model

    @staticmethod
    def preprocess_image_array(image_array):
        """Preprocess image array."""
        return vgg19.preprocess_input(image_array)


inception_v3_model = inception_v3.InceptionV3(weights="imagenet")
inception_v3_model = KerasModel(inception_v3_model.input, inception_v3_model.get_layer("avg_pool").output)
print('here inception_v3_model')


class ImageSearchInceptionV3(ImageSearch):
    """ImageSearch model using inception v3 for feature extraction."""

    @staticmethod
    def create_keras_model():
        """Get instance of the keras model being used."""
        return inception_v3_model

    @staticmethod
    def preprocess_image_array(image_array):
        """Preprocess image array."""
        return inception_v3.preprocess_input(image_array)


@pytest.fixture(params=["vgg16", "vgg19", "inception_v3"])
def image_search(app, request):
    """Fixture that returns an instance of image search."""
    app.config.update({"IMAGE_SEARCH_PATH": "image_search/" + request.param})
    request.getfixturevalue("db")

    if request.param == "vgg19":
        image_search = ImageSearchVGG19(app)
    elif request.param == "inception_v3":
        image_search = ImageSearchInceptionV3(app)
    elif request.param == "vgg16":
        image_search = ImageSearchVGG16(app)
    else:
        image_search = ImageSearch(app)

    yield image_search


@pytest.fixture
def Radio(db):
    """Radio db.Model for the deafult database."""

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
    """Image db.Model for the deafult database."""

    @image_search.register()
    class Image(db.Model):
        id = db.Column(db.Integer, primary_key=True)
        path = db.Column(db.String, nullable=False)
        radio_id = db.Column(db.Integer, db.ForeignKey(Radio.id), nullable=False)

    _load_fixture_vectors(image_search, Image)
    return Image
