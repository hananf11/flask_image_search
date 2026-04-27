import logging
import os
import threading
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
from flask_image_search.helper import derive_namespace

logger = logging.getLogger(__name__)

__all__ = (
    "ImageSearch",
    "ImageSearchError",
    __version__,
    __author__,
    __author_email__,
    __license__,
)


class ImageSearchError(Exception):
    """Raised for setup / configuration problems in Flask-Image-Search."""


class ImageSearch:
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
            raise ImageSearchError(
                "Flask-SQLAlchemy must be initialised on the app before Flask-Image-Search."
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
            or derive_namespace(self.model, self.feature_size)
        )

        if backend is None:
            # db.engine on Flask-SQLAlchemy 3.x requires an active app context;
            # push one ourselves so callers can construct ImageSearch at
            # module import time without manually wrapping the call.
            with app.app_context():
                backend = auto_backend(self.db.engine.dialect.name)
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
        # Prevents thread explosion on shared hosting (e.g. PythonAnywhere).
        torch.set_num_threads(1)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        backbone = torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT)
        backbone.classifier = backbone.classifier[:1]
        for param in backbone.parameters():
            param.grad = None
        backbone.to(self.device)
        backbone.train(False)
        return backbone

    def get_feature_size(self):
        """Return the dimensionality of the feature vectors produced by ``self.model``.

        Override if the probing forward-pass is too slow or unavailable.
        """
        test_input = torch.randn(1, 3, 224, 224).to(self.device)
        with torch.no_grad():
            return self.model(test_input).shape[1]

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
