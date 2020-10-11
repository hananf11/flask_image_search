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
    app.config.update({
        "IMAGE_SEARCH_PATH_PREFIX": "../resources/image_search_tests/"
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
def model_model(db):
    class Model(db.Model):
        id = db.Column(db.Integer, primary_key=True)
        name = db.Column(db.Text)

        images = db.relationship("Image")
    return Model


@pytest.fixture
def image_model(db, image_search, model_model):

    @image_search.register()
    class Image(db.Model):
        id = db.Column(db.Integer, primary_key=True)
        path_ = db.Column('path', db.String, nullable=False)
        model_id = db.Column(db.Integer, db.ForeignKey(model_model.id), nullable=False)

        @property
        def path(self):
            return os.path.join('../resources/', self.path_)

    image_search.index_model(Image)
    return Image


@pytest.mark.filterwarnings("ignore::DeprecationWarning:tensorflow")
def test_indexed(image_model, image_search):
    """Test that images can be removed from the index correctly and reindexed."""
    image_search.index_model(image_model)  # index any missing images
    original_features = image_search.features(image_model).copy()  # copy the features so they can be compared later

    # pick a random image to be removed from the index
    image_to_be_deleted = image_model.query.order_by(func.random()).first()
    image_search.delete_index(image_to_be_deleted)  # remove an index

    assert len(original_features) - 1 == len(image_search.features(image_model))

    image_search.index_model(image_model)

    assert len(original_features) == len(image_search.features(image_model))


def test_search(image_model, image_search):
    results = image_search.search(image_model, os.path.join(BASE_PATH, "../resources/test.jpg"), 5)
    # check that the results are correct by checking the ids
    assert [result[0] for result in results] == ['4512_439', '2649_439', '4514_371', '4516_371', '2194_438']


def test_query_search(image_model, image_search):
    images = image_model.query.image_search(os.path.join(BASE_PATH, "../resources/test.jpg"), 5).all()
    # check that the correct Images were returned
    assert [str(image.id) for image in images] == ['4512', '2649', '4514', '4516', '2194']


def test_transform_query_search(image_model, image_search):
    images = image_model.query.with_transformation(
        image_search.query_search(os.path.join(BASE_PATH, "../resources/test.jpg"), 5)
    ).all()
    # check that the correct Images were returned
    assert [str(image.id) for image in images] == ['4512', '2649', '4514', '4516', '2194']


def test_query_search_join(db, image_model, model_model, image_search):
    query = model_model.query.join(image_model).options(db.contains_eager(model_model.images))
    query = query.image_search(os.path.join(BASE_PATH, "../resources/test.jpg"), 3, join=True)
    models = query.all()
    assert [str(model.id) for model in models] == ['439', '371', '438']
    assert [str(image.id) for image in models[0].images] == ['4512', '2649', '2204', '4513', '5115', '5117', '5116']
    assert [str(image.id) for image in models[1].images] == ['4514', '4516',
                                                             '4517', '4518', '1798', '1799', '4515', '4519', '1800']
    assert [str(image.id) for image in models[2].images] == ['2194', '2197', '2196', '2193', '2195']
