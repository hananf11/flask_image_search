"""Tests for `flask_image_search` package."""

import logging
import os

import pytest
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.sql.expression import func

from flask_image_search import ImageSearch

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s Testing: %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

BASE_PATH = os.path.dirname(os.path.realpath(__file__))


@pytest.fixture
def app():
    """Fixture that returns an instance of FLask."""
    return Flask(__name__)


@pytest.fixture
def image_search(app):
    """Fixture that returns an instance of image search."""

    return ImageSearch(app)


@pytest.fixture
def db(app):
    app.config.update({
        "SQLALCHEMY_DATABASE_URI": "sqlite:///test.db",
        "SQLALCHEMY_TRACK_MODIFICATIONS": False
    })
    return SQLAlchemy(app)


@pytest.fixture
def radio_model(db):
    class Radio(db.Model):
        id = db.Column(db.Integer, primary_key=True)
        name = db.Column(db.Text)

        images = db.relationship("Image")
    return Radio


@pytest.fixture
def image_model(db, image_search, radio_model):

    @image_search.register()
    class Image(db.Model):
        id = db.Column(db.Integer, primary_key=True)
        path = db.Column(db.String, nullable=False)
        radio_id = db.Column(db.Integer, db.ForeignKey(radio_model.id), nullable=False)

    image_search.index_model(Image, threaded=False)
    return Image


@pytest.mark.filterwarnings("ignore::DeprecationWarning:tensorflow")
def test_indexed(image_model, image_search):
    """Test that images can be removed from the index correctly and reindexed."""
    original_features = image_search.features(image_model).copy()  # copy the features so they can be compared later

    # pick a random image to be removed from the index
    image_to_be_deleted = image_model.query.order_by(func.random()).first()
    image_search.delete_index(image_to_be_deleted)  # remove an index

    assert len(original_features) - 1 == len(image_search.features(image_model))

    image_search.index_model(image_model, threaded=False)

    assert len(original_features) == len(image_search.features(image_model))


def test_search(image_model, image_search):
    results = image_search.search(image_model, os.path.join(BASE_PATH, "./test.jpg"))
    # check that the results are correct by checking the ids
    assert [result[0] for result in results[:5]] == ['4512', '2649', '4514', '4516', '2194']


def test_query_search(image_model, image_search):
    images = image_model.query.image_search(os.path.join(BASE_PATH, "./test.jpg")).all()
    # check that the correct Images were returned
    assert [image.id for image in images[:5]] == [4512, 2649, 4514, 4516, 2194]


def test_transform_query_search(image_model, image_search):
    images = image_model.query.with_transformation(
        image_search.query_search(os.path.join(BASE_PATH, "./test.jpg"))
    ).all()
    # check that the correct Images were returned
    assert [image.id for image in images[:5]] == [4512, 2649, 4514, 4516, 2194]


def test_query_search_join(db, image_model, radio_model, image_search):
    query = radio_model.query.join(image_model).options(db.contains_eager(radio_model.images))
    query = query.image_search(os.path.join(BASE_PATH, "./test.jpg"), join=True)
    radios = query.all()[:3]
    for radio in radios:
        for image in radio.images:
            assert image.radio_id == radio.id
    assert [model.id for model in radios] == [439, 371, 438]
    assert [image.id for image in radios[0].images] == [4512, 2649, 2204, 4513, 5115, 5117, 5116]
    assert [image.id for image in radios[1].images] == [4514, 4516, 4517, 4518, 1798, 1799, 4515, 4519, 1800]
    assert [image.id for image in radios[2].images] == [2194, 2197, 2196, 2193, 2195]
