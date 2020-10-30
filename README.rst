==================
Flask Image Search
==================


.. image:: https://img.shields.io/pypi/v/flask_image_search.svg
        :target: https://pypi.python.org/pypi/flask_image_search

.. image:: https://img.shields.io/travis/com/hananf11/flask_image_search.svg
        :target: https://travis-ci.com/hananf11/flask_image_search

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

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
