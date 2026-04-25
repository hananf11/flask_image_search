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


Flask Image Search is an extension for Flask that adds content-based image
search to a Flask + Flask-SQLAlchemy app. Features are extracted with a
PyTorch CNN and stored alongside your model rows; queries are ranked by
L2 distance.

* Free software: MIT license
* Documentation: https://flask-image-search.readthedocs.io


Features
--------

* Works with `Flask-SQLAlchemy`_ — search your database directly.
* Indexes update automatically when rows are added, edited, or deleted.
* Pluggable storage backends: works on any SQL dialect; opts into
  ``sqlite-vec`` or ``pgvector`` when available for ANN-accelerated KNN.

.. _Flask-SQLAlchemy: https://flask-sqlalchemy.palletsprojects.com/

Install
-------

::

    pip install flask-image-search

See the `documentation <https://flask-image-search.readthedocs.io>`_ for
usage, backend options, and the migration guide from 1.x.

Development
-----------

Requires Python 3.9+ and pip 25.1+::

    pip install -e . --group dev
    pre-commit install --hook-type pre-commit --hook-type pre-push

Common tasks (``poe --help`` lists everything)::

    poe test          # run tests
    poe lint          # ruff check
    poe docs          # live-reload docs server
    poe example       # run the demo Flask app

Cutting a release::

    bump-my-version bump patch    # or: minor / major
    git push --tags               # CI publishes to PyPI

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
