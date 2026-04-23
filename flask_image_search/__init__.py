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
    case as sa_case,
    column as sa_column,
    event,
    literal_column,
    select,
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

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

__all__ = (
    "ImageSearch",
    "VectorBackend",
    "GenericBackend",
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
            matrix=None,
            pks=None,
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
        store.table.create(bind=engine, checkfirst=True)
        store.created = True

    def _invalidate(self, store):
        store.matrix = None
        store.pks = None

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
        self._invalidate(store)

    def delete(self, connection, model, pk):
        self._ensure_created(model.__tablename__)
        store = self._stores[model.__tablename__]
        connection.execute(store.table.delete().where(store.table.c.pk == pk))
        self._invalidate(store)

    def _load(self, model):
        store = self._stores[model.__tablename__]
        if store.matrix is not None:
            return store

        self._ensure_created(model.__tablename__)
        conn = store.image_search.db.session.connection()
        rows = conn.execute(
            select(store.table.c.pk, store.table.c.embedding).order_by(store.table.c.pk)
        ).all()

        if not rows:
            store.matrix = np.zeros((0, store.dim), dtype=np.float32)
            store.pks = []
            return store

        pks = [r[0] for r in rows]
        matrix = np.empty((len(rows), store.dim), dtype=np.float32)
        for i, (_, blob) in enumerate(rows):
            matrix[i] = np.frombuffer(blob, dtype=np.float32)

        store.matrix = matrix
        store.pks = pks
        return store

    def search(self, model, query_vector, sorted=True, limit=None):
        store = self._load(model)
        if store.matrix.shape[0] == 0:
            return ()

        query = np.asarray(query_vector, dtype=np.float32)
        distances = np.linalg.norm(store.matrix - query, axis=1)

        if sorted:
            order = np.argsort(distances, kind="stable")
        else:
            order = range(len(distances))

        results = tuple((store.pks[i], float(distances[i])) for i in order)
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
        return self._load(model).matrix.shape[0]

    def _is_indexed(self, connection, model, pk):
        store = self._stores[model.__tablename__]
        self._ensure_created(model.__tablename__)
        row = connection.execute(
            select(store.table.c.pk).where(store.table.c.pk == pk)
        ).first()
        return row is not None


def _sql_literal(value):
    if isinstance(value, (int, float)):
        return str(value)
    escaped = str(value).replace("'", "''")
    return f"'{escaped}'"


# --------------------------------------------------------------------------- #
# Backend dispatch
# --------------------------------------------------------------------------- #


def _auto_backend(dialect_name):
    """Pick the best available backend for ``dialect_name``.

    Phase 1 ships only :class:`GenericBackend`. SqliteVecBackend / PgVectorBackend
    slot in here without changing public API.
    """
    return GenericBackend()


_NAMESPACE_SAFE = re.compile(r"[^A-Za-z0-9_]+")


def _derive_namespace(keras_model):
    """Derive a stable, table-name-safe namespace from the given Keras model.

    Any change to architecture, output shape, or cut-layer flows through to a
    different namespace, so switching feature extractors never silently reuses
    stale vectors written by a prior extractor.
    """
    if keras_model is None:
        return "default"

    name = getattr(keras_model, "name", None) or "model"
    try:
        dim = keras_model.output_shape[-1]
    except Exception:
        dim = "x"

    try:
        config_repr = repr(keras_model.get_config())
    except Exception:
        config_repr = repr(keras_model)
    short = hashlib.sha1(config_repr.encode("utf-8")).hexdigest()[:8]

    raw = f"{name}_{dim}_{short}"
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

    __slots__ = ["root", "db", "keras_model", "models", "app", "backend", "_namespace"]

    def __init__(self, app=None, **kwargs):
        self.app = app
        if app is not None:
            self.init_app(app, **kwargs)

    def init_app(self, app, tensorflow=True, backend=None, namespace=None):
        self.root = app.root_path

        sqlalchemy = app.extensions.get("sqlalchemy")
        if sqlalchemy is None:
            raise Exception(
                "You need to initialize Flask-SQLAlchemy before Flask-Image-Search."
            )
        self.db = getattr(sqlalchemy, "db", sqlalchemy)

        self.keras_model = self.create_keras_model() if tensorflow else None
        self.models = {}
        self._namespace = (
            namespace
            or app.config.get("IMAGE_SEARCH_NAMESPACE")
            or _derive_namespace(self.keras_model)
        )

        if backend is None:
            backend = _auto_backend(self.db.engine.dialect.name)
        self.backend = backend

    @property
    def namespace(self):
        """Identifier used to scope the vector store to this feature extractor.

        Auto-derived from the Keras model (``<name>_<dim>_<config-hash>``) so
        swapping backbones can never cross-contaminate vector tables. Override
        by passing ``namespace=`` to :meth:`init_app` / ``__init__`` or by
        setting ``IMAGE_SEARCH_NAMESPACE`` in Flask config.
        """
        return self._namespace

    @staticmethod
    def create_keras_model():
        import keras
        base_model = keras.applications.vgg16.VGG16(weights="imagenet")
        return keras.Model(
            inputs=base_model.input, outputs=base_model.get_layer("fc1").output
        )

    @staticmethod
    def preprocess_image_array(image_array):
        from keras.applications.vgg16 import preprocess_input
        return preprocess_input(image_array)

    def feature_extract(self, image):
        from keras.preprocessing.image import img_to_array

        if self.keras_model:
            image_size = self.keras_model.input_shape[1:3]
            image = image.resize(image_size).convert("RGB")
            image_array = img_to_array(image)
            image_array = np.expand_dims(image_array, axis=0)
            input_array = self.preprocess_image_array(image_array)

            feature = self.keras_model.predict(input_array)[0]
            return feature / np.linalg.norm(feature)
        return np.random.rand(4096)

    def register(self, id="id", path="path", ignore="ignore"):
        def inner(model):
            if self.keras_model is not None:
                dim = self.keras_model.output_shape[-1]
            else:
                dim = 4096

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

        if not replace and isinstance(self.backend, GenericBackend):
            if self.backend._is_indexed(connection, model, pk):
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

    def _session_connection(self):
        """Return the caller's active connection.

        Routing writes through ``db.session.connection()`` means the vector
        write participates in whatever transaction the caller already owns —
        a Flask request handler's request-scoped session, a test fixture's
        rollback-wrapped transaction, or an explicit ``with db.session.begin()``
        block. A flush is required because the underlying SQLAlchemy events
        that wire vector writes to parent-row writes only fire on flush.
        """
        return self.db.session.connection()

    def index(self, entry, replace=False):
        result = self._on_upsert(self._session_connection(), entry, replace=replace)
        self.db.session.flush()
        return result

    def index_model(self, model, replace=False, threaded=True):
        def thread_content():
            entries = self.db.session.query(model).options(lazyload("*")).all()
            total = 0
            indexed = 0
            conn = self._session_connection()
            for entry in entries:
                total += 1
                if self._on_upsert(conn, entry, replace=replace):
                    indexed += 1
            self.db.session.flush()
            logger.info(
                f"Indexed {indexed} of {total} images for the model {model.__tablename__}"
            )

        if threaded:
            threading.Thread(target=thread_content).start()
        else:
            thread_content()

    def delete_index(self, entry):
        self._on_delete(self._session_connection(), entry)
        self.db.session.flush()

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
