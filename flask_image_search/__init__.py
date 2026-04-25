import hashlib
import logging
import os
import re
import threading
from types import SimpleNamespace

import numpy as np
from PIL import Image
from sqlalchemy import (
    Column,
    LargeBinary,
    Table,
    event,
    func,
    literal_column,
    select,
    text,
)
from sqlalchemy import (
    case as sa_case,
)
from sqlalchemy import (
    column as sa_column,
)
from sqlalchemy.orm import lazyload

from flask_image_search.__about__ import (
    __author__,
    __author_email__,
    __license__,
    __version__,
)

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s flask image search: %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)


__all__ = (
    "ImageSearch",
    "VectorBackend",
    "GenericBackend",
    "SqliteVecBackend",
    "PgVectorBackend",
    __version__,
    __author__,
    __author_email__,
    __license__,
)


# --------------------------------------------------------------------------- #
# Backend protocol
# --------------------------------------------------------------------------- #


class VectorBackend:
    """Storage backend for indexed feature vectors."""

    def register(self, image_search, model, dim):
        raise NotImplementedError

    def upsert(self, connection, model, pk, vector):
        raise NotImplementedError

    def delete(self, connection, model, pk):
        raise NotImplementedError

    def search(self, model, query_vector, sorted=True, limit=None):
        raise NotImplementedError

    def distance_expr(self, model, query_vector, column, limit=None):
        raise NotImplementedError

    def count_indexed(self, model):
        raise NotImplementedError

    def is_indexed(self, connection, model, pk):
        """Return True if pk already has a vector stored.

        Used by ImageSearch to skip re-indexing when replace=False. Backends
        that use INSERT OR REPLACE semantics can keep the default (False) and
        always let upsert run.
        """
        return False


# --------------------------------------------------------------------------- #
# GenericBackend: works on every dialect SQLAlchemy supports.
# --------------------------------------------------------------------------- #


class GenericBackend(VectorBackend):
    """Dialect-agnostic backend.

    Vectors are stored as raw float32 bytes in a LargeBinary column on a
    sibling table ``<tablename>_vectors__<namespace>(pk, embedding)``.
    Queries load the full corpus into a numpy matrix (cached per-process,
    invalidated on mutation) and run a brute-force L2 scan.
    """

    def __init__(self):
        self._stores = {}

    def register(self, image_search, model, dim):
        tablename = model.__tablename__
        pk_attr = image_search.models[tablename].id
        pk_col = getattr(model, pk_attr)
        pk_type = pk_col.property.columns[0].type.copy()

        metadata = model.metadata
        vec_tablename = f"{tablename}_vectors__{image_search.namespace}"

        if vec_tablename in metadata.tables:
            table = metadata.tables[vec_tablename]
        else:
            table = Table(
                vec_tablename,
                metadata,
                Column("pk", pk_type, primary_key=True),
                Column("embedding", LargeBinary, nullable=False),
            )

        self._stores[tablename] = SimpleNamespace(
            table=table,
            dim=dim,
            image_search=image_search,
            created=False,
        )
        self._ensure_created(tablename)

    # Rows per streaming chunk during search. At dim=4096 float32, a chunk of
    # 4096 holds ~64 MB — bounded regardless of corpus size. Override in a
    # subclass for constrained hosts (smaller) or fat corpora (larger).
    _CHUNK_SIZE = 4096

    def _ensure_created(self, tablename):
        store = self._stores[tablename]
        if store.created:
            return
        try:
            engine = store.image_search.db.engine
        except Exception:
            return
        store.table.create(bind=engine, checkfirst=True)
        store.created = True

    def _encode(self, vector, dim):
        arr = np.asarray(vector, dtype=np.float32)
        if arr.shape != (dim,):
            raise ValueError(f"expected vector of shape ({dim},), got {arr.shape}")
        return arr.tobytes()

    def upsert(self, connection, model, pk, vector):
        self._ensure_created(model.__tablename__)
        store = self._stores[model.__tablename__]
        blob = self._encode(vector, store.dim)
        table = store.table
        connection.execute(table.delete().where(table.c.pk == pk))
        connection.execute(table.insert().values(pk=pk, embedding=blob))

    def delete(self, connection, model, pk):
        self._ensure_created(model.__tablename__)
        store = self._stores[model.__tablename__]
        connection.execute(store.table.delete().where(store.table.c.pk == pk))

    def search(self, model, query_vector, sorted=True, limit=None):
        self._ensure_created(model.__tablename__)
        store = self._stores[model.__tablename__]
        query = np.asarray(query_vector, dtype=np.float32)

        pks = []
        dists = []
        chunk = np.empty((self._CHUNK_SIZE, store.dim), dtype=np.float32)

        with store.image_search.db.engine.connect() as conn:
            result = conn.execute(
                select(store.table.c.pk, store.table.c.embedding).order_by(store.table.c.pk)
            )
            for rows in iter(lambda: result.fetchmany(self._CHUNK_SIZE), []):
                n = len(rows)
                buf = chunk[:n] if n == self._CHUNK_SIZE else np.empty((n, store.dim), dtype=np.float32)
                for i, (_, blob) in enumerate(rows):
                    buf[i] = np.frombuffer(blob, dtype=np.float32)
                chunk_dists = np.linalg.norm(buf - query, axis=1)
                pks.extend(r[0] for r in rows)
                dists.extend(chunk_dists.tolist())

        if not pks:
            return ()

        if sorted:
            order = np.argsort(np.asarray(dists), kind="stable")
        else:
            order = range(len(dists))

        results = tuple((pks[i], float(dists[i])) for i in order)
        if limit is not None:
            results = results[:limit]
        return results

    def distance_expr(self, model, query_vector, column, limit=None):
        results = self.search(model, query_vector, sorted=limit is not None, limit=limit)

        if isinstance(column, str):
            column = sa_column(column)

        whens = []
        for pk, distance in results:
            if limit and len(whens) >= limit:
                break
            whens.append(
                (column == literal_column(_sql_literal(pk)), literal_column(str(distance)))
            )

        if not whens:
            return None
        return sa_case(*whens, else_=literal_column("9999"))

    def count_indexed(self, model):
        self._ensure_created(model.__tablename__)
        store = self._stores[model.__tablename__]
        with store.image_search.db.engine.connect() as conn:
            return conn.execute(select(func.count()).select_from(store.table)).scalar() or 0

    def is_indexed(self, connection, model, pk):
        store = self._stores[model.__tablename__]
        self._ensure_created(model.__tablename__)
        return connection.execute(
            select(store.table.c.pk).where(store.table.c.pk == pk)
        ).first() is not None


def _sql_literal(value):
    if isinstance(value, (int, float)):
        return str(value)
    escaped = str(value).replace("'", "''")
    return f"'{escaped}'"


# --------------------------------------------------------------------------- #
# SqliteVecBackend: uses sqlite-vec extension for in-DB vector search
# --------------------------------------------------------------------------- #


class SqliteVecBackend(VectorBackend):
    """SQLite backend using the ``sqlite-vec`` extension.

    Requires ``pip install sqlite-vec`` and a Python sqlite3 built with
    ``SQLITE_ENABLE_LOAD_EXTENSION`` (not available on all distributions —
    check with ``hasattr(sqlite3.connect(':memory:'), 'enable_load_extension')``).
    Vectors are stored in a ``vec0`` virtual table
    ``<tablename>_vec__<namespace>(pk, embedding float[dim])``.
    KNN search is delegated entirely to SQLite — no numpy matrix is loaded
    into memory.
    """

    def __init__(self):
        self._stores = {}
        self._engines_patched = set()

    @staticmethod
    def available():
        """Return True if sqlite-vec can be loaded on this Python build."""
        import sqlite3

        import sqlite_vec
        con = sqlite3.connect(":memory:")
        if not hasattr(con, "enable_load_extension"):
            return False
        try:
            con.enable_load_extension(True)
            sqlite_vec.load(con)
            return True
        except Exception:
            return False
        finally:
            con.close()

    def _patch_engine(self, engine):
        if id(engine) in self._engines_patched:
            return
        import sqlite_vec

        @event.listens_for(engine, "connect")
        def _load_ext(dbapi_conn, _record):
            dbapi_conn.enable_load_extension(True)
            sqlite_vec.load(dbapi_conn)
            dbapi_conn.enable_load_extension(False)

        self._engines_patched.add(id(engine))

    def register(self, image_search, model, dim):
        tablename = model.__tablename__
        pk_attr = image_search.models[tablename].id
        pk_col = getattr(model, pk_attr)
        pk_type = pk_col.property.columns[0].type.copy()

        self._stores[tablename] = SimpleNamespace(
            dim=dim,
            vec_tablename=f"{tablename}_vec__{image_search.namespace}",
            image_search=image_search,
            pk_type=pk_type,
            created=False,
        )
        self._ensure_created(tablename)

    def _ensure_created(self, tablename):
        store = self._stores[tablename]
        if store.created:
            return
        try:
            engine = store.image_search.db.engine
        except Exception:
            return
        self._patch_engine(engine)
        conn = store.image_search.db.session.connection()
        conn.execute(text(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {store.vec_tablename} "
            f"USING vec0(pk INTEGER PRIMARY KEY, embedding float[{store.dim}])"
        ))
        store.created = True

    def _serialize(self, vector):
        import sqlite_vec
        return sqlite_vec.serialize_float32(np.asarray(vector, dtype=np.float32).tolist())

    def upsert(self, connection, model, pk, vector):
        self._ensure_created(model.__tablename__)
        store = self._stores[model.__tablename__]
        blob = self._serialize(vector)
        connection.execute(
            text(f"DELETE FROM {store.vec_tablename} WHERE pk = :pk"), {"pk": pk}
        )
        connection.execute(
            text(f"INSERT INTO {store.vec_tablename}(pk, embedding) VALUES (:pk, :emb)"),
            {"pk": pk, "emb": blob},
        )

    def delete(self, connection, model, pk):
        self._ensure_created(model.__tablename__)
        store = self._stores[model.__tablename__]
        connection.execute(
            text(f"DELETE FROM {store.vec_tablename} WHERE pk = :pk"), {"pk": pk}
        )

    def search(self, model, query_vector, sorted=True, limit=None):
        self._ensure_created(model.__tablename__)
        store = self._stores[model.__tablename__]
        blob = self._serialize(query_vector)
        conn = store.image_search.db.session.connection()

        if limit is not None:
            # Fast KNN path — sqlite-vec uses its ANN index. `distance` here is
            # squared L2; sqrt it on the way out so results match GenericBackend
            # and PgVectorBackend (true L2). Ordering stays on raw `distance`
            # since sqrt is monotonic — same ranking, cheaper sort.
            rows = conn.execute(
                text(
                    f"SELECT pk, sqrt(distance) FROM {store.vec_tablename} "
                    f"WHERE embedding MATCH :q ORDER BY distance LIMIT :lim"
                ),
                {"q": blob, "lim": limit},
            ).fetchall()
        else:
            order_clause = "ORDER BY distance" if sorted else ""
            rows = conn.execute(
                text(
                    f"SELECT pk, vec_distance_L2(embedding, :q) AS distance "
                    f"FROM {store.vec_tablename} {order_clause}"
                ),
                {"q": blob},
            ).fetchall()

        return tuple((row[0], float(row[1])) for row in rows)

    def distance_expr(self, model, query_vector, column, limit=None):
        results = self.search(model, query_vector, sorted=limit is not None, limit=limit)

        if isinstance(column, str):
            column = sa_column(column)

        whens = []
        for pk, distance in results:
            if limit and len(whens) >= limit:
                break
            whens.append(
                (column == literal_column(_sql_literal(pk)), literal_column(str(distance)))
            )

        if not whens:
            return None
        return sa_case(*whens, else_=literal_column("9999"))

    def count_indexed(self, model):
        self._ensure_created(model.__tablename__)
        store = self._stores[model.__tablename__]
        conn = store.image_search.db.session.connection()
        row = conn.execute(text(f"SELECT COUNT(*) FROM {store.vec_tablename}")).fetchone()
        return row[0]


# --------------------------------------------------------------------------- #
# PgVectorBackend: uses pgvector extension for PostgreSQL
# --------------------------------------------------------------------------- #


class PgVectorBackend(VectorBackend):
    """PostgreSQL backend using the ``pgvector`` extension.

    Requires ``pip install pgvector`` and the ``vector`` extension installed
    in the database (``CREATE EXTENSION IF NOT EXISTS vector``).  Vectors are
    stored in a sibling table ``<tablename>_vectors__<namespace>(pk, embedding vector(dim))``.
    Distance ordering uses pgvector's ``<->`` L2 operator, which benefits from
    IVFFlat / HNSW indexes on large corpora.
    """

    def __init__(self):
        self._stores = {}

    def register(self, image_search, model, dim):
        from pgvector.sqlalchemy import Vector

        tablename = model.__tablename__
        pk_attr = image_search.models[tablename].id
        pk_col = getattr(model, pk_attr)
        pk_type = pk_col.property.columns[0].type.copy()

        metadata = model.metadata
        vec_tablename = f"{tablename}_vectors__{image_search.namespace}"

        if vec_tablename in metadata.tables:
            table = metadata.tables[vec_tablename]
        else:
            table = Table(
                vec_tablename,
                metadata,
                Column("pk", pk_type, primary_key=True),
                Column("embedding", Vector(dim), nullable=False),
            )

        self._stores[tablename] = SimpleNamespace(
            table=table,
            dim=dim,
            image_search=image_search,
            created=False,
        )
        self._ensure_created(tablename)

    def _ensure_created(self, tablename):
        store = self._stores[tablename]
        if store.created:
            return
        try:
            engine = store.image_search.db.engine
        except Exception:
            return
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
        store.table.create(bind=engine, checkfirst=True)
        store.created = True

    def upsert(self, connection, model, pk, vector):
        self._ensure_created(model.__tablename__)
        store = self._stores[model.__tablename__]
        vec = np.asarray(vector, dtype=np.float32).tolist()
        table = store.table
        connection.execute(table.delete().where(table.c.pk == pk))
        connection.execute(table.insert().values(pk=pk, embedding=vec))

    def delete(self, connection, model, pk):
        self._ensure_created(model.__tablename__)
        store = self._stores[model.__tablename__]
        table = store.table
        connection.execute(table.delete().where(table.c.pk == pk))

    def search(self, model, query_vector, sorted=True, limit=None):
        self._ensure_created(model.__tablename__)
        store = self._stores[model.__tablename__]
        vec = np.asarray(query_vector, dtype=np.float32).tolist()
        table = store.table

        dist_expr = table.c.embedding.op("<->")(text(f"'{vec}'::vector"))
        q = select(table.c.pk, dist_expr.label("distance"))
        if sorted:
            q = q.order_by("distance")
        if limit is not None:
            q = q.limit(limit)

        conn = store.image_search.db.session.connection()
        rows = conn.execute(q).fetchall()
        return tuple((row[0], float(row[1])) for row in rows)

    def distance_expr(self, model, query_vector, column, limit=None):
        self._ensure_created(model.__tablename__)
        store = self._stores[model.__tablename__]
        vec = np.asarray(query_vector, dtype=np.float32).tolist()
        table = store.table

        if isinstance(column, str):
            column = sa_column(column)

        # Correlated subquery: evaluated per-row by PostgreSQL, benefits from
        # pgvector index when used with LIMIT in the outer query.
        dist = (
            select(table.c.embedding.op("<->")(text(f"'{vec}'::vector")))
            .where(table.c.pk == column)
            .scalar_subquery()
        )

        if limit is not None:
            # Fall back to CASE to honour the limit contract
            results = self.search(model, query_vector, sorted=True, limit=limit)
            whens = [
                (column == literal_column(_sql_literal(pk)), literal_column(str(distance)))
                for pk, distance in results
            ]
            if not whens:
                return None
            return sa_case(*whens, else_=literal_column("9999"))

        return dist

    def count_indexed(self, model):
        from sqlalchemy import func
        self._ensure_created(model.__tablename__)
        store = self._stores[model.__tablename__]
        conn = store.image_search.db.session.connection()
        row = conn.execute(select(func.count()).select_from(store.table)).fetchone()
        return row[0]


# --------------------------------------------------------------------------- #
# Backend dispatch
# --------------------------------------------------------------------------- #


def _auto_backend(dialect_name):
    """Pick the best available backend for ``dialect_name``."""
    if dialect_name == "sqlite":
        try:
            import sqlite_vec  # noqa: F401
            if SqliteVecBackend.available():
                return SqliteVecBackend()
        except ImportError:
            pass
    elif dialect_name == "postgresql":
        try:
            from pgvector.sqlalchemy import Vector  # noqa: F401
            return PgVectorBackend()
        except ImportError:
            pass
    return GenericBackend()


_NAMESPACE_SAFE = re.compile(r"[^A-Za-z0-9_]+")


def _derive_namespace(model, dim=None):
    """Derive a stable, table-name-safe namespace from a torch.nn.Module.

    Any change to architecture or output dimension flows through to a different
    namespace, so switching backbones never silently reuses stale vectors.
    ``str(model)`` produces a stable layer-dump that changes with architecture.
    """
    if model is None:
        return "default"

    name = model.__class__.__name__.lower()
    short = hashlib.sha1(str(model).encode("utf-8")).hexdigest()[:8]
    raw = f"{name}_{dim or 'x'}_{short}"
    return _NAMESPACE_SAFE.sub("_", raw).strip("_") or "default"


# --------------------------------------------------------------------------- #
# ImageSearch
# --------------------------------------------------------------------------- #


class ImageSearch(object):
    """Flask extension adding content-based image search.

    .. code-block:: python

        app = Flask(__name__)
        db = SQLAlchemy(app)
        image_search = ImageSearch(app)
    """

    __slots__ = ["root", "db", "model", "device", "feature_size", "models", "app", "backend", "_namespace",
                 "preprocess"]

    def __init__(self, app=None, **kwargs):
        self.app = app
        if app is not None:
            self.init_app(app, **kwargs)

    def init_app(self, app, load_model=True, backend=None, namespace=None):
        self.app = app
        self.root = app.root_path

        sqlalchemy = app.extensions.get("sqlalchemy")
        if sqlalchemy is None:
            raise Exception(
                "You need to initialize Flask-SQLAlchemy before Flask-Image-Search."
            )
        self.db = getattr(sqlalchemy, "db", sqlalchemy)

        self.device = None
        self.model = self.get_model() if load_model else None
        self.feature_size = self.get_feature_size() if self.model is not None else 4096
        self.preprocess = self.get_preprocess() if self.model is not None else None
        self.models = {}
        self._namespace = (
            namespace
            or app.config.get("IMAGE_SEARCH_NAMESPACE")
            or _derive_namespace(self.model, self.feature_size)
        )

        if backend is None:
            # db.engine on Flask-SQLAlchemy 3.x requires an active app context;
            # push one ourselves so callers can construct ImageSearch at
            # module import time without manually wrapping the call.
            with app.app_context():
                backend = _auto_backend(self.db.engine.dialect.name)
        self.backend = backend

    @property
    def namespace(self):
        """Identifier used to scope the vector store to this feature extractor.

        Auto-derived from the model's architecture and output dimension so
        swapping backbones can never cross-contaminate vector tables. Override
        by passing ``namespace=`` to :meth:`init_app` / ``__init__`` or by
        setting ``IMAGE_SEARCH_NAMESPACE`` in Flask config.
        """
        return self._namespace

    def get_model(self):
        """Load and return the feature extraction model.

        Override in a subclass to use a different backbone. Also responsible
        for setting ``self.device`` and switching the model to inference mode.

        Default: torchvision VGG16 with ImageNet weights, classifier
        truncated to the first FC layer (4096-d output).
        """
        import torch
        import torchvision

        # Prevents thread explosion on shared hosting (e.g. PythonAnywhere).
        torch.set_num_threads(1)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        backbone = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT)
        backbone.classifier = backbone.classifier[:1]
        for param in backbone.parameters():
            param.grad = None
        backbone.to(self.device)
        backbone.train(False)  # inference mode — equivalent to .eval()
        return backbone

    def get_feature_size(self):
        """Return the dimensionality of the feature vectors produced by ``self.model``.

        Override if the probing forward-pass is too slow or unavailable.
        """
        import torch
        test_input = torch.randn(1, 3, 224, 224).to(self.device)
        with torch.no_grad():
            return self.model(test_input).shape[1]

    def get_preprocess(self):
        """Return the preprocessing transform for images before the forward pass.

        Defaults to the transform recommended by the weights used in ``get_model()``.
        Override when using a custom backbone with different input requirements.
        Must accept a PIL image and return a float tensor of shape ``(C, H, W)``.
        """
        import torchvision.models as M

        return M.VGG16_Weights.DEFAULT.transforms()

    def feature_extract(self, image):
        """Extract an L2-normalised feature vector from a PIL image.

        Override to use a fully custom extraction pipeline.
        Must return a 1-D float32 numpy array of length ``self.feature_size``.
        """
        import torch

        if self.model is None:
            return np.random.rand(self.feature_size)

        tensor = self.preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feature = self.model(tensor)[0].cpu().numpy()
        return feature / np.linalg.norm(feature)

    def register(self, id="id", path="path", ignore="ignore"):
        def inner(model):
            dim = self.feature_size

            self.models[model.__tablename__] = SimpleNamespace(
                id=id,
                path=path,
                ignore=ignore if ignore and hasattr(model, ignore) else False,
                dim=dim,
            )

            self.backend.register(self, model, dim)

            @event.listens_for(model, "after_delete")
            def _deleted(mapper, connection, target):
                self._on_delete(connection, target)

            @event.listens_for(model, "after_insert")
            def _inserted(mapper, connection, target):
                self._on_upsert(connection, target, replace=False)

            @event.listens_for(model, "after_update")
            def _updated(mapper, connection, target):
                self._on_upsert(connection, target, replace=True)

            try:
                count = self.backend.count_indexed(model)
                logger.info(f"Loaded {count} image features for '{model.__tablename__}'")
            except Exception:
                pass

            return model

        return inner

    def _on_upsert(self, connection, entry, replace):
        data = self.models[entry.__tablename__]
        if data.ignore and getattr(entry, data.ignore):
            return False

        pk = getattr(entry, data.id)
        model = type(entry)

        if not replace and self.backend.is_indexed(connection, model, pk):
            return True

        image_path = getattr(entry, data.path)
        image = Image.open(os.path.join(self.root, image_path))
        features = self.feature_extract(image)

        self.backend.upsert(connection, model, pk, features)
        return True

    def _on_delete(self, connection, entry):
        model = type(entry)
        pk = getattr(entry, self.models[entry.__tablename__].id)
        self.backend.delete(connection, model, pk)

    def index(self, entry, replace=False):
        with self.db.engine.begin() as conn:
            return self._on_upsert(conn, entry, replace=replace)

    # Commit every N entries during index_model. Small enough to keep the SQLite
    # write lock window brief (letting readers interleave), large enough to
    # amortise transaction overhead over many inserts.
    _INDEX_BATCH_SIZE = 50

    def index_model(self, model, replace=False, threaded=True):
        def run():
            with self.app.app_context():
                entries = self.db.session.query(model).options(lazyload("*")).all()
                total = 0
                indexed = 0
                batch = []

                def flush():
                    nonlocal indexed
                    if not batch:
                        return
                    with self.db.engine.begin() as conn:
                        for entry in batch:
                            if self._on_upsert(conn, entry, replace=replace):
                                indexed += 1
                    batch.clear()

                for entry in entries:
                    total += 1
                    batch.append(entry)
                    if len(batch) >= self._INDEX_BATCH_SIZE:
                        flush()
                flush()

                logger.info(
                    f"Indexed {indexed} of {total} images for the model {model.__tablename__}"
                )

        if threaded:
            threading.Thread(target=run).start()
        else:
            run()

    def delete_index(self, entry):
        with self.db.engine.begin() as conn:
            self._on_delete(conn, entry)

    def _resolve_query_image(self, image):
        if isinstance(image, str):
            return Image.open(os.path.join(self.root, image))
        return image

    def search(self, model, image, sorted=True, limit=None):
        if isinstance(model, str):
            raise TypeError(
                "search(model=...) must be a SQLAlchemy model class; passing a "
                "table name string is no longer supported in 2.x."
            )
        image = self._resolve_query_image(image)
        query = self.feature_extract(image)
        return self.backend.search(model, query, sorted=sorted, limit=limit)

    def case(self, image, model, column=None, limit=None):
        image = self._resolve_query_image(image)
        query = self.feature_extract(image)

        if column is None:
            column = getattr(model, self.models[model.__tablename__].id)

        return self.backend.distance_expr(model, query, column, limit=limit)

    def count_indexed(self, model):
        return self.backend.count_indexed(model)
