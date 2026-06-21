import logging
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torchvision
from PIL import Image
from sqlalchemy import event
from sqlalchemy.orm import lazyload
from torchvision.models import VGG16_Weights

from flask_image_search.__about__ import (
    __author__,
    __author_email__,
    __license__,
    __version__,
)
from flask_image_search.backends import auto_backend
from flask_image_search.helper import extract_features, extract_features_batch

logger = logging.getLogger(__name__)

__all__ = (
    "ImageSearch",
    "ImageSearchError",
    "__author__",
    "__author_email__",
    "__license__",
    "__version__",
)


class ImageSearchError(Exception):
    """Raised for setup / configuration problems in Flask-Image-Search."""


class InvalidSearchModelError(TypeError):
    """search() received a string instead of a SQLAlchemy model."""


class ImageSearch:
    """Flask extension adding content-based image search.

    .. code-block:: python

        app = Flask(__name__)
        db = SQLAlchemy(app)
        image_search = ImageSearch(app)
    """

    #: Dimensionality of the feature vectors produced by ``self.model``.
    #: Must match the real model output. Override on subclasses.
    feature_size = 4096

    #: Identifier used to scope the vector store to this feature extractor.
    #: Must be unique per (model architecture, feature_size). Renaming this
    #: invalidates every previously indexed vector, so pick once and keep it.
    namespace = "vgg16-fc1"

    def __init__(self, app=None, **kwargs):
        self.app = app
        if app is not None:
            self.init_app(app, **kwargs)

    def init_app(self, app, load_model=True, backend=None, namespace=None):
        self.app = app
        self.root = app.root_path

        sqlalchemy = app.extensions.get("sqlalchemy")
        if sqlalchemy is None:
            raise ImageSearchError(
                "Flask-SQLAlchemy must be initialised on the app before Flask-Image-Search."
            )
        self.db = getattr(sqlalchemy, "db", sqlalchemy)

        ns = (
            namespace
            or app.config.get("IMAGE_SEARCH_NAMESPACE")
            or type(self).namespace
        )
        if not ns:
            raise ImageSearchError(
                "ImageSearch.namespace must be set (class attr, init_app(namespace=...), "
                "or IMAGE_SEARCH_NAMESPACE config)."
            )
        self.namespace = ns

        self.device = None
        self.model = None
        self.preprocess = None
        if load_model:
            self.model = self.get_model()  # also sets self.device
            self.preprocess = self.get_preprocess()
        self.models = {}

        if backend is None:
            # db.engine on Flask-SQLAlchemy 3.x requires an active app context;
            # push one ourselves so callers can construct ImageSearch at
            # module import time without manually wrapping the call.
            with app.app_context():
                backend = auto_backend(self.db.engine.dialect.name)
        self.backend = backend

    def get_model(self):
        """Load and return the feature extraction model.

        Override in a subclass to use a different backbone. Also responsible
        for setting ``self.device`` and switching the model to inference mode.

        Default: torchvision VGG16 with ImageNet weights, classifier
        truncated to the first FC layer (4096-d output).
        """
        # Prevents thread explosion on shared hosting (e.g. PythonAnywhere).
        torch.set_num_threads(1)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        backbone = torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT)
        backbone.classifier = backbone.classifier[:1]
        backbone.to(self.device)
        backbone.train(False)
        return backbone

    def get_preprocess(self):
        """Return the preprocessing transform for images before the forward pass.

        Defaults to the transform recommended by the weights used in ``get_model()``.
        Override when using a custom backbone with different input requirements.
        Must accept a PIL image and return a float tensor of shape ``(C, H, W)``.
        """
        return VGG16_Weights.DEFAULT.transforms()

    def feature_extract(self, image):
        """Extract an L2-normalised feature vector from a PIL image.

        Override to use a fully custom extraction pipeline.
        Must return a 1-D float32 numpy array of length ``self.feature_size``.
        """
        if self.model is None:
            return np.random.rand(self.feature_size)
        return extract_features(self.model, self.preprocess, image, self.device)

    def feature_extract_batch(self, images):
        """Extract L2-normalised vectors for a list of PIL images in one pass.

        Used by ``index_model`` to amortise inference over many images. The
        default runs a single batched forward through ``self.model`` -- several
        times faster than per-image extraction on CPU, far more on GPU.

        If a subclass overrides ``feature_extract`` (a fully custom single-image
        pipeline) but not this method, falls back to calling ``feature_extract``
        per image so custom behaviour is never silently bypassed. Override this
        method too to batch a custom pipeline.
        """
        if self.model is None:
            return [np.random.rand(self.feature_size) for _ in images]
        if type(self).feature_extract is ImageSearch.feature_extract:
            return extract_features_batch(self.model, self.preprocess, images, self.device)
        return [self.feature_extract(image) for image in images]

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
            def _deleted(_mapper, connection, target):
                self.delete_index(target, connection=connection)

            @event.listens_for(model, "after_insert")
            def _inserted(_mapper, connection, target):
                self.index(target, replace=False, connection=connection)

            @event.listens_for(model, "after_update")
            def _updated(_mapper, connection, target):
                self.index(target, replace=True, connection=connection)

            return model

        return inner

    def index(self, entry, replace=False, connection=None):
        if connection is None:
            with self.db.engine.begin() as conn:
                return self.index(entry, replace=replace, connection=conn)

        data = self.models[entry.__tablename__]
        if data.ignore and getattr(entry, data.ignore):
            return False

        pk = getattr(entry, data.id)
        model = type(entry)

        if not replace and self.backend.is_indexed(connection, model, pk):
            return True

        image_path = getattr(entry, data.path)
        if not image_path:
            logger.warning(
                "Skipping %s pk=%s: %r is empty/None",
                entry.__tablename__, pk, data.path,
            )
            return False
        image = Image.open(Path(self.root) / image_path)
        features = self.feature_extract(image)

        self.backend.upsert(connection, model, pk, features)
        return True

    def _iter_indexable(self, model, entries, read_conn, replace):
        """Yield ``(pk, loaded PIL image)`` for each entry that should be indexed.

        Applies the same per-entry guards as ``index``: skip ignored rows,
        already-indexed rows (unless ``replace``), empty paths, and images that
        fail to open -- so one bad row never aborts an ``index_model`` run.
        """
        data = self.models[model.__tablename__]
        for entry in entries:
            if data.ignore and getattr(entry, data.ignore):
                continue
            pk = getattr(entry, data.id)
            if not replace and self.backend.is_indexed(read_conn, model, pk):
                continue
            image_path = getattr(entry, data.path)
            if not image_path:
                logger.warning(
                    "Skipping %s pk=%s: %r is empty/None",
                    model.__tablename__, pk, data.path,
                )
                continue
            try:
                image = Image.open(Path(self.root) / image_path)
                image.load()
            except Exception as exc:
                logger.warning(
                    "Skipping %s pk=%s: cannot open %s (%s)",
                    model.__tablename__, pk, image_path, exc,
                )
                continue
            yield pk, image

    def index_model(self, model, replace=False, threaded=True, batch_size=16):
        # batch_size controls both inference batching (one forward per batch)
        # and the SQLite write-lock window (one transaction per batch). 16 is
        # near the CPU throughput plateau while keeping each commit brief.
        def run():
            with self.app.app_context():
                entries = self.db.session.query(model).options(lazyload("*")).all()
                total = len(entries)
                indexed = 0
                pending = []  # (pk, PIL image) awaiting a batched forward pass

                def flush():
                    nonlocal indexed
                    if not pending:
                        return
                    vectors = self.feature_extract_batch([img for _, img in pending])
                    with self.db.engine.begin() as conn:
                        for (pk, _), vector in zip(pending, vectors):
                            self.backend.upsert(conn, model, pk, vector)
                            indexed += 1
                    pending.clear()

                with self.db.engine.connect() as read_conn:
                    for pk, image in self._iter_indexable(model, entries, read_conn, replace):
                        pending.append((pk, image))
                        if len(pending) >= batch_size:
                            flush()
                flush()

                logger.info(
                    f"Indexed {indexed} of {total} images for the model {model.__tablename__}"
                )

        if threaded:
            threading.Thread(target=run).start()
        else:
            run()

    def delete_index(self, entry, connection=None):
        if connection is None:
            with self.db.engine.begin() as conn:
                self.delete_index(entry, connection=conn)
            return
        model = type(entry)
        pk = getattr(entry, self.models[entry.__tablename__].id)
        self.backend.delete(connection, model, pk)

    def _resolve_query_image(self, image):
        if isinstance(image, str):
            return Image.open(Path(self.root) / image)
        return image

    def search(self, model, image, sorted=True, limit=None):
        if isinstance(model, str):
            raise InvalidSearchModelError(
                "search(model=...) must be a SQLAlchemy model class; "
                "passing a table name string is no longer supported in 2.x."
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
