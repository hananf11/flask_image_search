"""Tests for `flask_image_search` package."""

import logging

import pytest
import os
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
    app.config.update({
        "IMAGE_SEARCH_PATH_PREFIX": "../resources/image_search/"
    })

    return ImageSearch(app)


@pytest.fixture
def db(app):
    app.config.update({
        "SQLALCHEMY_DATABASE_URI": f"sqlite:///{app.root_path}/../resources/test.db",
        "SQLALCHEMY_TRACK_MODIFICATIONS": False
    })
    return SQLAlchemy(app)


@pytest.fixture
def test_table(db, image_search):

    @image_search.register()
    class Image(db.Model):
        id = db.Column(db.Integer, primary_key=True)
        url = db.Column(db.String, nullable=False)
        model_id = db.Column(db.Integer, nullable=False)

    image_search.index_model(Image)
    return Image


@pytest.mark.filterwarnings("ignore::DeprecationWarning:tensorflow")
def test_indexed(test_table, image_search):
    """Test that images can be removed from the index correctly and reindexed."""
    image_search.index_model(test_table)  # index any missing images
    original_features = image_search.features(test_table).copy()  # copy the features so they can be compared later

    # pick a random image to be removed from the index
    image_to_be_deleted = test_table.query.order_by(func.random()).first()
    image_search.delete_index(image_to_be_deleted)  # remove an index

    assert len(original_features) - 1 == len(image_search.features(test_table))

    image_search.index_model(test_table)

    assert len(original_features) == len(image_search.features(test_table))


def test_search(test_table, image_search):
    results = image_search.search(test_table, os.path.join(BASE_PATH, "../resources/test.jpg"), 5)
    # check that the results are correct by checking the ids
    assert [result[0] for result in results] == ['4512', '2649', '4514', '4516', '2194']


def test_query_search(test_table, image_search):
    images = test_table.query.image_search(os.path.join(BASE_PATH, "../resources/test.jpg"), 5).all()
    # check that the correct Images were returned
    assert [str(image.id) for image in images] == ['4512', '2649', '4514', '4516', '2194']


def test_transform_query_search(test_table, image_search):
    images = test_table.query.with_transformation(
        image_search.query_search(os.path.join(BASE_PATH, "../resources/test.jpg"), 5)
    ).all()
    # check that the correct Images were returned
    assert [str(image.id) for image in images] == ['4512', '2649', '4514', '4516', '2194']
