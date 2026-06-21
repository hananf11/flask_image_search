"""End-to-end test with provable ranking.

The model-backed tests assert opaque golden ID lists from real photos -- they
catch drift but can't say whether a change made results better or worse. This
file swaps the CNN for a deterministic stand-in (embedding = per-channel mean
RGB) and feeds solid-colour PIL images, so a solid colour maps to its own RGB
and L2 distance is just colour distance. The correct ordering is therefore
known by construction, and it exercises the real public chain end to end:
register -> auto-index on insert -> search / case() -> delete.
"""

import numpy as np
import pytest
from flask import Flask
from flask_image_search import ImageSearch
from flask_sqlalchemy import SQLAlchemy
from PIL import Image as PILImage

# Colours chosen so L2 distance from pure red strictly increases (no ties for a
# backend to break arbitrarily): red 0, orange ~0.50, yellow 1.0, blue ~1.41.
COLORS = {
    "red": (255, 0, 0),
    "orange": (255, 128, 0),
    "yellow": (255, 255, 0),
    "blue": (0, 0, 255),
}
ORDER_FROM_RED = ["red", "orange", "yellow", "blue"]


class MeanRGBSearch(ImageSearch):
    """Deterministic stand-in for the CNN: embedding = per-channel mean / 255.

    A solid-colour image maps to its own normalised RGB, so ranking by L2
    distance is ranking by colour distance -- provable, no weights, no
    randomness. Overrides the batch path too; the base one returns random
    vectors when ``self.model is None`` (load_model=False here).
    """

    feature_size = 3
    namespace = "mean-rgb-e2e"

    def feature_extract(self, image):
        arr = np.asarray(image.convert("RGB"), dtype=np.float32)
        return arr.reshape(-1, 3).mean(axis=0) / 255.0

    def feature_extract_batch(self, images):
        return [self.feature_extract(img) for img in images]


def _query(color):
    """A solid-colour PIL image to search with (passed straight to search)."""
    return PILImage.new("RGB", (4, 4), COLORS[color])


@pytest.fixture
def e2e(tmp_path):
    """Yield (image_search, Image, ids) with four colour rows indexed.

    Rows are inserted in a scrambled order so the assertions prove ranking, not
    insertion order. Insertion alone indexes them -- via the after_insert event
    listener register() attaches -- which is part of what this exercises.
    """
    app = Flask(__name__)
    app.config.update(
        SQLALCHEMY_DATABASE_URI=f"sqlite:///{tmp_path / 'e2e.db'}",
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
    )
    db = SQLAlchemy(app)

    with app.app_context():
        image_search = MeanRGBSearch(app, load_model=False)

        @image_search.register()
        class Image(db.Model):
            id = db.Column(db.Integer, primary_key=True)
            # Absolute path -- index() joins it to app.root_path, and an absolute
            # path wins that join, so the on-disk PNG is found regardless of root.
            path = db.Column(db.String, nullable=False)

        db.create_all()

        ids = {}
        for name in ["blue", "yellow", "red", "orange"]:
            png = tmp_path / f"{name}.png"
            PILImage.new("RGB", (8, 8), COLORS[name]).save(png)
            row = Image(path=str(png))
            db.session.add(row)
            db.session.commit()  # after_insert -> auto-index
            ids[name] = row.id

        yield image_search, Image, ids


def test_auto_index_on_insert(e2e):
    image_search, Image, _ids = e2e
    # inserting the rows (and nothing else) indexed all four
    assert image_search.count_indexed(Image) == len(COLORS)


def test_search_ranks_by_colour(e2e):
    image_search, Image, ids = e2e
    results = image_search.search(Image, _query("red"))
    assert [pk for pk, _ in results] == [ids[n] for n in ORDER_FROM_RED]
    # nearest is an exact colour match -> distance 0
    assert results[0][1] == pytest.approx(0.0, abs=1e-6)


def test_search_limit(e2e):
    image_search, Image, ids = e2e
    results = image_search.search(Image, _query("red"), limit=2)
    assert [pk for pk, _ in results] == [ids["red"], ids["orange"]]


def test_case_orders_query(e2e):
    image_search, Image, ids = e2e
    rows = Image.query.order_by(image_search.case(_query("red"), Image)).all()
    assert [r.id for r in rows] == [ids[n] for n in ORDER_FROM_RED]


def test_delete_removes_from_results(e2e):
    image_search, Image, ids = e2e
    db = image_search.db
    db.session.delete(db.session.get(Image, ids["blue"]))
    db.session.commit()  # after_delete -> drop the vector

    assert image_search.count_indexed(Image) == len(COLORS) - 1
    results = image_search.search(Image, _query("red"))
    assert ids["blue"] not in [pk for pk, _ in results]
