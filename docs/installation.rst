Installation
============

From PyPI
---------

.. code-block:: console

    $ pip install flask-image-search

ANN-accelerated backends are opt-in extras:

.. code-block:: console

    $ pip install flask-image-search[sqlite]    # sqlite-vec
    $ pip install flask-image-search[postgres]  # pgvector

Without an extra, the dialect-agnostic ``GenericBackend`` is used.

From source
-----------

.. code-block:: console

    $ git clone https://github.com/hananf11/flask_image_search
    $ cd flask_image_search
    $ pip install -e .
