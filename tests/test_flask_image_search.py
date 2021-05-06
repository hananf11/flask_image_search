"""Tests for `flask_image_search` package."""

import logging
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # noqa

import pytest
import zarr
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from keras.applications import inception_v3, vgg19
from keras.models import Model as KerasModel
from sqlalchemy.sql.expression import func

from flask_image_search import ImageSearch

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s Testing: %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
IMAGE = os.path.join(BASE_PATH, "./test.jpg")


@pytest.fixture(name="app")
def fixture_app():
    """Fixture that returns an instance of FLask."""
    return Flask(__name__)


@pytest.fixture(name="db")
def fixture_flask_sqlalchemy(app):
    app.config.update({
        "SQLALCHEMY_DATABASE_URI": "sqlite:///test.db",
        "SQLALCHEMY_TRACK_MODIFICATIONS": False
    })
    return SQLAlchemy(app)


@pytest.fixture(name="image_search")
def fixture_image_search(app, request):
    """Fixture that returns an instance of image search."""
    params = getattr(request, "param", {})

    class MyImageSearch(ImageSearch):
        if params.get("model"):
            @staticmethod
            def create_keras_model():
                base_model = params["model"]
                base_model = base_model(weights="imagenet")

                inputs = base_model.input
                outputs = base_model.get_layer(params["out"]).output
                return KerasModel(inputs=inputs, outputs=outputs)

            @staticmethod
            def preprocess_image_array(image_array):
                return params["preprocess"](image_array)

    app.config.update({"IMAGE_SEARCH_PATH": params.get("storage", "image_search/vgg16")})

    return MyImageSearch(app)


@pytest.fixture(name="Radio")
def fixture_Radio(db):

    class Radio(db.Model):
        id = db.Column(db.Integer, primary_key=True)
        name = db.Column(db.Text)

        images = db.relationship("Image")
    return Radio


@pytest.fixture(name="Image")
def fixture_Image(db, image_search, Radio):

    @image_search.register()
    class Image(db.Model):
        id = db.Column(db.Integer, primary_key=True)
        path = db.Column(db.String, nullable=False)
        radio_id = db.Column(db.Integer, db.ForeignKey(Radio.id), nullable=False)

    image_search.index_model(Image, threaded=False)
    return Image


vgg19_test_case = {
    "storage": "image_search/vgg19",
    "model": vgg19.VGG19,
    "preprocess": vgg19.preprocess_input,
    "out": "fc1"
}

inception_v3_test_case = {
    "storage": "image_search/inception_v3",
    "model": inception_v3.InceptionV3,
    "preprocess": inception_v3.preprocess_input,
    "out": "avg_pool"
}


@pytest.mark.filterwarnings("ignore::DeprecationWarning:tensorflow")
@pytest.mark.parametrize("image_search", [
    {},
    vgg19_test_case,
    inception_v3_test_case,
], indirect=["image_search"])
def test_index_image(Image, image_search):
    """Test that indexing images is working correctly"""
    if "/image_backup" in image_search.storage:
        del image_search.storage["/image_backup"]
    zarr.copy_all(
        image_search.storage["/image_features"],
        image_search.storage.require_group("/image_backup")
    )
    print(type(image_search.features(Image)))

    # choose a random image to delete from the image index
    image_to_be_deleted = Image.query.order_by(func.random()).first()
    image_search.delete_index(image_to_be_deleted)

    # check that the image was removed from the index
    assert len(image_search.storage["/image_backup"]) - 1 == len(image_search.storage["/image_features"])

    image_search.index_model(Image, threaded=False)  # index all missing images

    # check the features extracted haven't changed
    image_features = image_search.storage["/image_features"]
    image_features_backup = image_search.storage["/image_backup"]
    assert [image_features_backup[key] == image_features[key] for key in image_features]


@pytest.mark.parametrize("image_search, expected", [
    ({}, ["4512", "2649", "4514", "4516", "2194"]),
    (vgg19_test_case, ["2649", "4512", "4514", "2197", "4516"]),
    (inception_v3_test_case, ["4512", "4516", "4514", "5171", "2649"]),
], indirect=["image_search"])
def test_search(Image, image_search, expected):
    results = image_search.search(Image, IMAGE)
    # check that the results are correct by checking the ids
    assert [result[0] for result in results[:5]] == expected


@pytest.mark.parametrize("image_search, expected", [
    ({}, [4512, 2649, 4514, 4516, 2194]),
    (vgg19_test_case, [2649, 4512, 4514, 2197, 4516]),
    (inception_v3_test_case, [4512, 4516, 4514, 5171, 2649]),
], indirect=["image_search"])
def test_query_search(Image, image_search, expected):
    # images = Image.query.image_search().all()
    images = Image.query.order_by(image_search.case(IMAGE, Image)).limit(5)
    # check that the correct Images were returned
    assert [image.id for image in images] == expected


@pytest.mark.parametrize("image_search, expected_radios, expected_images_1, expected_images_2, expected_images_3", [
    (
        {},
        [439, 371, 438],
        [4512, 2649, 2204, 4513, 5115, 5117, 5116],
        [4514, 4516, 4517, 4518, 1798, 1799, 4515, 4519, 1800],
        [2194, 2197, 2196, 2193, 2195]
    ),
    (
        vgg19_test_case,
        [439, 371, 438],
        [2649, 4512, 2204, 4513, 5115, 5117, 5116],
        [4514, 4516, 4517, 4518, 4515, 1798, 4519, 1799, 1800],
        [2197, 2194, 2196, 2195, 2193]
    ),
    (
        inception_v3_test_case,
        [439, 371, 1011],
        [4512, 2649, 2204, 4513, 5116, 5117, 5115],
        [4516, 4514, 1798, 4519, 4515, 1800, 4518, 4517, 1799],
        [5171, 5172, 5170, 5173, 5178, 5180, 5177, 5179, 5175, 5176, 5174, 5181]
    ),
], indirect=["image_search"])
def test_query_join_search(db, image_search, Image, Radio, expected_radios,
                           expected_images_1, expected_images_2, expected_images_3):
    query = Radio.query
    query = query.join(Image).options(db.contains_eager(Radio.images))  # join to images
    query = query.order_by(image_search.case(IMAGE, Image))
    radios = query.all()[:3]

    assert [model.id for model in radios] == expected_radios
    assert [image.id for image in radios[0].images] == expected_images_1
    assert [image.id for image in radios[1].images] == expected_images_2
    assert [image.id for image in radios[2].images] == expected_images_3
