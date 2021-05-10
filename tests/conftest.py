import logging
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # noqa

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
def db(app):
    """Fixture for sqlalchemy"""
    app.config.update({
        "SQLALCHEMY_DATABASE_URI": "sqlite:///test.db",
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


@pytest.fixture
def Image(db, image_search, Radio):
    """Image db.Model for the deafult database."""

    @image_search.register()
    class Image(db.Model):
        id = db.Column(db.Integer, primary_key=True)
        path = db.Column(db.String, nullable=False)
        radio_id = db.Column(db.Integer, db.ForeignKey(Radio.id), nullable=False)

    image_search.index_model(Image, threaded=False)
    return Image
