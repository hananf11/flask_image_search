"""Tests for `flask_image_search` package."""

import logging
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # noqa

import pytest
import zarr
from sqlalchemy.sql.expression import func

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s Testing: %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

BASE_PATH = os.path.dirname(os.path.realpath(__file__))
IMAGE = os.path.join(BASE_PATH, "./test.jpg")


@pytest.mark.filterwarnings("ignore::DeprecationWarning:tensorflow")
@pytest.mark.parametrize(
    "image_search",
    ["default", "vgg16", "vgg19", "inception_v3"],
    ids=["default", "vgg16", "vgg19", "inception_v3"],
    indirect=["image_search"]
)
def test_index_image(Image, image_search, tmp_path):
    """Test that indexing images is working correctly"""
    tmp_storage = zarr.open(str(tmp_path / 'tmp.zarr'), mode='a')
    zarr.copy_all(
        image_search.storage["/image_features"],
        tmp_storage
    )

    # choose a random image to delete from the image index
    image_to_be_deleted = Image.query.order_by(func.random()).first()
    image_search.delete_index(image_to_be_deleted)

    # check that the image was removed from the index
    assert len(tmp_storage) - 1 == len(image_search.storage["/image_features"])

    image_search.index_model(Image, threaded=False)  # index all missing images

    # check the features extracted haven't changed
    image_features = image_search.storage["/image_features"]
    assert [tmp_storage[key] == image_features[key] for key in image_features]


@pytest.mark.parametrize(
    "image_search, expected",
    [
        ("vgg16", ["4512", "2649", "4514", "4516", "2194"]),
        ("vgg19", ["2649", "4512", "4514", "2197", "4516"]),
        ("inception_v3", ["4512", "4516", "4514", "5171", "2649"]),
    ],
    ids=["vgg16", "vgg19", "inception_v3"],
    indirect=["image_search"],


)
def test_search(Image, image_search, expected):
    results = image_search.search(Image, IMAGE)
    # check that the results are correct by checking the ids
    assert [result[0] for result in results[:5]] == expected


@pytest.mark.parametrize(
    "image_search, expected",
    [
        ("vgg16", [4512, 2649, 4514, 4516, 2194]),
        ("vgg19", [2649, 4512, 4514, 2197, 4516]),
        ("inception_v3", [4512, 4516, 4514, 5171, 2649]),
    ],
    ids=["vgg16", "vgg19", "inception_v3"],
    indirect=["image_search"]
)
def test_query_search(Image, image_search, expected):
    # images = Image.query.image_search().all()
    images = Image.query.order_by(image_search.case(IMAGE, Image)).limit(5)
    # check that the correct Images were returned
    assert [image.id for image in images] == expected


@pytest.mark.parametrize(
    "image_search, expected",
    [
        (
            "vgg16",
            {
                439: [4512, 2649, 2204, 4513, 5115, 5117, 5116],
                371: [4514, 4516, 4517, 4518, 1798, 1799, 4515, 4519, 1800],
                438: [2194, 2197, 2196, 2193, 2195]
            }
        ),
        (
            "vgg19",
            {
                439: [2649, 4512, 2204, 4513, 5115, 5117, 5116],
                371: [4514, 4516, 4517, 4518, 4515, 1798, 4519, 1799, 1800],
                438: [2197, 2194, 2196, 2195, 2193]

            }
        ),
        (
            "inception_v3",
            {
                439: [4512, 2649, 2204, 4513, 5116, 5117, 5115],
                371: [4516, 4514, 1798, 4519, 4515, 1800, 4518, 4517, 1799],
                1011: [5171, 5172, 5170, 5173, 5178, 5180, 5177, 5179, 5175, 5176, 5174, 5181]
            }
        ),
    ],
    ids=["vgg16", "vgg19", "inception_v3"],
    indirect=["image_search"]
)
def test_query_join_search(db, image_search, Image, Radio, expected):
    query = Radio.query
    query = query.join(Image).options(db.contains_eager(Radio.images))  # join to images
    query = query.order_by(image_search.case(IMAGE, Image))
    radios = query.all()[:3]

    assert [model.id for model in radios] == list(expected.keys())
    assert [image.id for image in radios[0].images] == expected[radios[0].id]
    assert [image.id for image in radios[1].images] == expected[radios[1].id]
    assert [image.id for image in radios[2].images] == expected[radios[2].id]
