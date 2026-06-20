"""Backend contract tests.

Exercise every VectorBackend (GenericBackend, SqliteVecBackend, PgVectorBackend)
through the same hand-built vectors so ranking is provable by construction --
no CNN, no test.db, fully deterministic.

* generic    -- always runs (pure NumPy, every dialect).
* sqlite_vec -- skipped unless the sqlite-vec extension loads on this build.
* pgvector   -- skipped unless ``pgvector`` is installed AND
                ``FIS_TEST_POSTGRES_URI`` points at a Postgres with the vector
                extension available (the CI ``backends`` job provides one).
"""

import os
from uuid import uuid4

import numpy as np
import pytest
from flask import Flask
from flask_image_search import ImageSearch
from flask_image_search.backends import (
    GenericBackend,
    PgVectorBackend,
    SqliteVecBackend,
)
from flask_sqlalchemy import SQLAlchemy

DIM = 4

# query == V1, so distance ordering is provable: 0 < ~0.14 < ~1.41
QUERY = [1.0, 0.0, 0.0, 0.0]
VECTORS = {
    1: [1.0, 0.0, 0.0, 0.0],   # identical -> distance 0
    2: [0.0, 1.0, 0.0, 0.0],   # orthogonal -> farthest
    3: [0.9, 0.1, 0.0, 0.0],   # near V1 -> middle
}
EXPECTED_ORDER = [1, 3, 2]


def _unavailable(reason):
    """Skip locally, but hard-fail when FIS_REQUIRE_BACKENDS is set.

    CI sets FIS_REQUIRE_BACKENDS so a backend that silently skips (missing
    extension, no Postgres) turns the PR red instead of green -- the whole
    point of the backends job is that all three actually run.
    """
    if os.environ.get("FIS_REQUIRE_BACKENDS"):
        pytest.fail(f"FIS_REQUIRE_BACKENDS set but backend unavailable: {reason}")
    pytest.skip(reason)


def _make_backend(kind, tmp_path):
    """Return (backend, sqlalchemy_uri) for ``kind`` or skip if unavailable."""
    if kind == "generic":
        return GenericBackend(), f"sqlite:///{tmp_path / 'backend.db'}"
    if kind == "sqlite_vec":
        if not SqliteVecBackend.available():
            _unavailable("sqlite-vec extension cannot load on this Python build")
        return SqliteVecBackend(), f"sqlite:///{tmp_path / 'backend.db'}"
    if kind == "pgvector":
        uri = os.environ.get("FIS_TEST_POSTGRES_URI")
        if not uri:
            _unavailable("FIS_TEST_POSTGRES_URI not set")
        if not PgVectorBackend.available():
            _unavailable("pgvector not installed")
        return PgVectorBackend(), uri
    raise ValueError(kind)


@pytest.fixture(params=["generic", "sqlite_vec", "pgvector"])
def backend_case(request, tmp_path):
    """Yield (image_search, Item) wired to one backend, inside an app context.

    A unique namespace + table name per case keeps a shared (CI) Postgres from
    colliding across parametrizations and runs.
    """
    backend, uri = _make_backend(request.param, tmp_path)

    app = Flask(__name__)
    app.config.update(
        SQLALCHEMY_DATABASE_URI=uri,
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
    )
    db = SQLAlchemy(app)

    suffix = uuid4().hex[:8]

    with app.app_context():
        class Item(db.Model):
            __tablename__ = f"item_{suffix}"
            id = db.Column(db.Integer, primary_key=True)
            path = db.Column(db.String)

        class _ImageSearch(ImageSearch):
            feature_size = DIM
            namespace = f"test-{request.param}-{suffix}"

        image_search = _ImageSearch(app, load_model=False, backend=backend)
        image_search.register()(Item)

        yield image_search, Item


def _seed(image_search, Item, vectors=VECTORS):
    with image_search.db.engine.begin() as conn:
        for pk, vec in vectors.items():
            image_search.backend.upsert(conn, Item, pk, np.asarray(vec, dtype=np.float32))


def test_search_orders_by_distance(backend_case):
    image_search, Item = backend_case
    _seed(image_search, Item)

    results = image_search.backend.search(Item, QUERY, limit=3)

    assert [pk for pk, _ in results] == EXPECTED_ORDER
    # the identical vector is at distance 0; distances are non-decreasing
    dists = [d for _, d in results]
    assert dists[0] == pytest.approx(0.0, abs=1e-5)
    assert dists == sorted(dists)


def test_search_limit_truncates(backend_case):
    image_search, Item = backend_case
    _seed(image_search, Item)

    results = image_search.backend.search(Item, QUERY, limit=2)

    assert [pk for pk, _ in results] == EXPECTED_ORDER[:2]


def test_search_unlimited_returns_all(backend_case):
    image_search, Item = backend_case
    _seed(image_search, Item)

    results = image_search.backend.search(Item, QUERY, sorted=True)

    assert len(results) == len(VECTORS)
    assert [pk for pk, _ in results] == EXPECTED_ORDER


def test_count_indexed(backend_case):
    image_search, Item = backend_case
    assert image_search.count_indexed(Item) == 0
    _seed(image_search, Item)
    assert image_search.count_indexed(Item) == len(VECTORS)


def test_upsert_replaces_in_place(backend_case):
    image_search, Item = backend_case
    _seed(image_search, Item)

    # move pk3 far away; it must drop to last without changing the row count
    with image_search.db.engine.begin() as conn:
        image_search.backend.upsert(conn, Item, 3, np.asarray([0.0, 0.0, 1.0, 0.0], dtype=np.float32))

    assert image_search.count_indexed(Item) == len(VECTORS)
    results = image_search.backend.search(Item, QUERY, limit=3)
    assert [pk for pk, _ in results] == [1, 2, 3]


def test_delete_removes_vector(backend_case):
    image_search, Item = backend_case
    _seed(image_search, Item)

    with image_search.db.engine.begin() as conn:
        image_search.backend.delete(conn, Item, 1)

    assert image_search.count_indexed(Item) == len(VECTORS) - 1
    results = image_search.backend.search(Item, QUERY, limit=3)
    assert 1 not in [pk for pk, _ in results]
    assert results[0][0] == 3  # nearest remaining vector
