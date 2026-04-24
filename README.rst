==================
Flask Image Search
==================


.. image:: https://img.shields.io/pypi/v/flask_image_search.svg
        :target: https://pypi.python.org/pypi/flask_image_search

.. image:: https://github.com/hananf11/flask_image_search/actions/workflows/action-ci-cd.yml/badge.svg
        :target: https://github.com/hananf11/flask_image_search/actions/workflows/action-ci-cd.yml
        :alt: CI/CD Build status

.. image:: https://readthedocs.org/projects/flask-image-search/badge/?version=latest
        :target: https://flask-image-search.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status




Flask Image Search is an extension for flask that makes it easy to add image searching to your flask site.


* Free software: MIT license
* Documentation: https://flask-image-search.readthedocs.io.


Features
--------

* Works with `Flask-SQLAlchemy`_ to make searching your database easy.
* Searching is fast
* The indexed images take up very little storage
* Database changes automatically update the image indexes

.. _Flask-SQLAlchemy: https://flask-sqlalchemy.palletsprojects.com/

How it works
------------

Flask-Image-Search uses a `CNN (Convolutional neural network) <https://en.wikipedia.org/wiki/Convolutional_neural_network>`_,
the CNN is used to extract the features from the images in a dataset. The features are stored so they can be searched later with a query image.
When an image is searched it's features are extracted and compared with all the other images to get the distance from the search image,
the distances are then used to sort the images.

Vector Storage Backends
-----------------------

Flask-Image-Search automatically picks the best available backend for your database.

.. list-table::
   :header-rows: 1
   :widths: 20 25 55

   * - Backend
     - Selected when
     - Notes
   * - ``GenericBackend``
     - Always (fallback)
     - Works on every database Flask-SQLAlchemy supports. Loads all vectors
       into a NumPy matrix in RAM on first query; brute-force L2 scan is
       O(N). Fast up to ~50 K images; memory grows linearly (10 K VGG16
       vectors ≈ 160 MB RAM).
   * - ``SqliteVecBackend``
     - SQLite + ``sqlite-vec`` installed + extension loading supported
     - Vectors stay in SQLite; KNN queries use sqlite-vec's ANN index (≈
       O(log N)). No corpus loaded into Python memory. Falls back to
       ``GenericBackend`` automatically if the extension cannot be loaded.
   * - ``PgVectorBackend``
     - PostgreSQL + ``pgvector`` installed
     - Native ``vector`` column type; ``<->`` L2 operator benefits from
       HNSW / IVFFlat indexes. Correlated-subquery ordering integrates
       cleanly with arbitrary SQLAlchemy queries.

**Checking SQLite extension support**

``SqliteVecBackend`` requires Python's ``sqlite3`` to be compiled with
``SQLITE_ENABLE_LOAD_EXTENSION``. Ubuntu, Debian, and macOS Python builds
include this; Fedora / RHEL builds typically do not. Check with::

    python3 -c "import sqlite3; c=sqlite3.connect(':memory:'); print(hasattr(c,'enable_load_extension'))"

If this prints ``False``, install ``pysqlite3-binary`` and configure
SQLAlchemy to use it, or use ``GenericBackend`` (the automatic fallback).

**Installing optional backends**::

    pip install flask-image-search[sqlite]    # SqliteVecBackend
    pip install flask-image-search[postgres]  # PgVectorBackend

**Choosing a backend explicitly**::

    from flask_image_search import ImageSearch, SqliteVecBackend

    image_search = ImageSearch(app, backend=SqliteVecBackend())

Development
-----------

**Setup**::

    pip install -e ".[dev]"
    pre-commit install --hook-type pre-commit --hook-type pre-push

This installs the package in editable mode with all dev tools, and wires up
two git hooks: ruff runs on every commit, the full pytest suite runs on every
push so broken code can't reach the remote.

**Common commands**::

    pytest                            # run tests
    ruff check flask_image_search tests  # lint

**Cutting a release**::

    # 1. Add a HISTORY.rst entry describing the changes
    # 2. Bump the version (updates __about__.py, commits, and tags automatically)
    bump-my-version bump patch        # or: minor / major
    git push --tags                   # CI builds, publishes to PyPI, creates GitHub release

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
