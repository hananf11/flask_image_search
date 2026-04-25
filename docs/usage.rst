.. module:: flask_image_search
    :noindex:


=====
Usage
=====

Initializing
------------

To use Flask Image Search in a project::

    from flask import Flask
    from flask_sqlachemy import SQLAlchemy
    from flask_image_search import ImageSearch


    app = Flask(__name__)
    db = SQLAlchemy(app)  # SQLAlchemy must be initialized first
    image_search = ImageSearch(app)

Alternatively you if you're using a `factory`_::

    db = SQLAlchemy()
    image_search = ImageSearch()

    def create_app():
        app = Flask(__name__)
        db.init_app(app)
        image_search.init_app(app)
        return app

.. _factory: https://flask.palletsprojects.com/en/1.1.x/patterns/appfactories/#basic-factories

.. note::
    Flask-Image-Search loads the PyTorch model on initialization, which can
    slow down the Flask dev-server restart loop. Disable model loading with::

        image_search = ImageSearch(app, load_model=False)

    When the model is disabled, :meth:`~ImageSearch.feature_extract` returns
    random vectors, so search results will be meaningless.

Config
------

+----------------------------+-------------------------------------------------------------------------+------------------+
| Option                     | Description                                                             | Default          |
+============================+=========================================================================+==================+
| ``IMAGE_SEARCH_NAMESPACE`` | Scopes the vector table to this feature extractor. Auto-derived from    | (auto-derived)   |
|                            | the model architecture hash so switching backbones never reuses         |                  |
|                            | stale vectors. Override only if you need a stable, human-readable name. |                  |
+----------------------------+-------------------------------------------------------------------------+------------------+

Registering Models
------------------

On any :class:`flask_sqlalchemy.Model` that you want to index you will need to register.
Registering a Model can be done using the :meth:`ImageSearch.register()` decorator.

To register a Model::

    @image_search.register()
    class Image(db.Model):
        id = db.Column(db.Integer, primary_key=True)
        path = db.Column(db.Text)

Ignoring columns
^^^^^^^^^^^^^^^^

if you want to ignore an image it can be done with an ignore column::

    @image_search.register()
    class Image(db.Model):
        ...
        ignore = db.Column(db.Boolean)

.. note::
    if you dont want to or cant use the default column names you can specify your own in :meth:`ImageSearch.register()`::

        @image_search.register(id='uid', path='url', ignore='exclude')
        class Image(db.Model):
            uid = db.Column(db.Integer, primary_key=True)
            url = db.Column(db.Text)
            exclude = db.Column(db.Boolean)

.. note::
    The attributes used by the register function do not need to be a :class:`~sqlalchemy.schema.Column`,
    this means you can use proprties to create the value, for example modifying the url stored in the database to make it absolute::

        @image_search.register()
        class Image(db.Model):
            ...
            url = db.Column(db.Text)

            @property
            def path(self):
                return os.path.join("/absolute_path/", self.url)

Indexing
--------

Once a Model has been registered the images must be indexed. Images that are not indexed will not be searched.

To index one image of a registered Model::

    image = Image.query.first()  # get an image however you like

    image_search.index(image)

To index all images of a registered Model::

    image_search.index_model(Image)

.. note::

    index_model is run insude a thread so that it does not stop your flask app.
    `threaded` can be set to `False` to stop it runing in a new thread.

Once Images are indexed changes on the database will be reflected on the index.
The indexes will be updated when an image is dropped, updated or a new image is added.

It is possible to manually delete an image from the index::

    image = Image.query.first()

    image_search.delete_index(image)


Making a query
--------------

Flask-Image-Search has a :meth:`~ImageSearch.case()` method that returns a :class:`~sqlalchemy.sql.expression.Case` that matches the distance from the image to the correct id,
this can be used to order a query.

Basic query
^^^^^^^^^^^

Here is an example of a simple query::

    case_statement = image_search.case("./image_path/image.png", Image)
    images = Image.query.order_by(case_statement).all()

Join query
^^^^^^^^^^

Because the distance is given as a case statement you can construct more advanced queries with joins to your image Model::

    class Animals(db.Model):
        id = db.Column(db.Integer, primary_key=True)
        name = db.Column(db.Text)

        images = db.relationship("Image")

    @image_search.register()
    class Image(db.Model):
        ...
        animal_id = db.Column(db.Integer, db.ForeignKey("animal.id"))

    case_statement = image_search.case("./image_path/image.png", Image)
    animals = Animals.query.join(Animals.images).options(db.contains_eager(Animals.images) \
              .order_by(case_statement)

Query with distance
^^^^^^^^^^^^^^^^^^^

Heres how to get the distance as a mapped attribute on your Model::

    @image_search.register()
    class Image(db.Model):
        ...

        distance = db.query_expression()

    case_statement = image_search.case("./image_path/image.png", Image).label("distance")
    images = Image.query.options(db.with_expression(Image.distance, case_statement)) \
             .order_by("distance").all()

Vector Storage Backends
-----------------------

Flask-Image-Search automatically selects a vector storage backend based on
your database dialect and what is installed:

- **SQLite + sqlite-vec** → :class:`SqliteVecBackend` — KNN search runs inside
  SQLite; no corpus loaded into Python memory. Requires ``pip install flask-image-search[sqlite]``
  and a Python build with ``SQLITE_ENABLE_LOAD_EXTENSION`` (Ubuntu, macOS;
  **not** Fedora/RHEL). Falls back to :class:`GenericBackend` automatically.
- **PostgreSQL + pgvector** → :class:`PgVectorBackend` — native ``vector``
  column type with ``<->`` L2 operator; benefits from HNSW/IVFFlat indexes.
  Requires ``pip install flask-image-search[postgres]``.
- **Everything else** → :class:`GenericBackend` — works on every dialect.
  Stores BLOBs in a sibling SQL table; loads all vectors into a NumPy matrix
  in RAM and does a brute-force L2 scan (O(N)). Fast up to ~50 K images;
  memory grows linearly (10 K VGG16 vectors ≈ 160 MB RAM).

To check whether SQLite extension loading is available on your system::

    python3 -c "import sqlite3; c=sqlite3.connect(':memory:'); print(hasattr(c,'enable_load_extension'))"

To choose a backend explicitly instead of relying on auto-selection::

    from flask_image_search import ImageSearch, SqliteVecBackend

    image_search = ImageSearch(app, backend=SqliteVecBackend())

Advanced
--------

Changing the backbone model
^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default Flask-Image-Search uses `torchvision VGG16`_ (4096-d, ImageNet
weights) and the preprocessing pipeline that ships with those weights.
Subclass :class:`ImageSearch` and override the relevant hooks to swap to
any ``torch.nn.Module``::

    import torchvision
    from torch import nn
    from flask_image_search import ImageSearch


    class InceptionSearch(ImageSearch):
        def get_model(self):
            import torch
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            m = torchvision.models.inception_v3(
                weights=torchvision.models.Inception_V3_Weights.DEFAULT
            )
            m.fc = nn.Identity()   # 2048-d output from the final pooling layer
            m.aux_logits = False
            m.to(self.device)
            m.train(False)
            return m

        def get_feature_size(self):
            return 2048

        def get_preprocess(self):
            return torchvision.models.Inception_V3_Weights.DEFAULT.transforms()

The hooks:

- :meth:`~ImageSearch.get_model` — return the ``torch.nn.Module`` and set
  ``self.device``.
- :meth:`~ImageSearch.get_feature_size` — output dimensionality.
- :meth:`~ImageSearch.get_preprocess` — the PIL→tensor transform. The
  default uses ``VGG16_Weights.DEFAULT.transforms()``, which handles
  resize, crop and normalise. Each torchvision weights enum exposes a
  matching ``.transforms()`` so you rarely need to write one by hand.

.. note::
    Switching backbones invalidates existing indexed vectors. The
    namespace is derived from the model architecture, so a fresh vector
    table is created automatically — but you must re-index::

        image_search.index_model(Image)

.. _torchvision VGG16: https://pytorch.org/vision/stable/models/generated/torchvision.models.vgg16.html
