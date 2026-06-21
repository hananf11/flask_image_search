Flask-Image-Search
==================

A `Flask`_ extension that adds content-based image search to a Flask +
`Flask-SQLAlchemy`_ app. Features are extracted with a PyTorch CNN
(default: torchvision VGG16) and stored alongside your model rows;
queries are ranked by L2 distance.

.. _Flask: https://flask.palletsprojects.com/
.. _Flask-SQLAlchemy: https://flask-sqlalchemy.palletsprojects.com/

User's Guide
------------

.. toctree::
    :maxdepth: 2

    installation
    usage

API Reference
-------------

.. toctree::
    :maxdepth: 2

    flask_image_search

Additional Notes
----------------

.. toctree::
    :maxdepth: 1

    history
    contributing
    authors
