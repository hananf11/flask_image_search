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
    flask_image_search loads keras/tensorflow when it is initialized,
    this can become a real pain when debugging your flask app,
    to stop tensorflow from loading and annoying you::

        image_search = ImageSearch(app, tensorflow=False)

    when tensorflow is disabled the image search will return random results.

Config
------

+---------------------------------+------------------------------------------------------------------------------------------------------------+--------------------+
| Option                          | Description                                                                                                | Default            |
+=================================+============================================================================================================+====================+
| ``IMAGE_SEARCH_PATH_PREFIX``    | This is a prefix that is added to the model ``__tablename__`` to get the file path for the index file.     | ``image_search/``  |
+---------------------------------+------------------------------------------------------------------------------------------------------------+--------------------+

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

    image_search.delete(Image)


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

Advanced
--------

Changing Keras Model
^^^^^^^^^^^^^^^^^^^^

By default `flask_image_search` uses the `VGG16`_ for it's feature extraction.
You can change the keras model used for feature extraction by overriding some of the :class:`ImageSearch` class methods.
Here is an example using `InceptionV3`_::

    from flask_image_search import ImageSearch
    from keras.applications.inception_v3 import InceptionV3, preprocess_input
    from keras.models import Model as KerasModel


    class MyImageSearch(ImageSearch):
        @staticmethod
        def create_keras_model():
            base_model = InceptionV3(weights="imagenet")
            return KerasModel(inputs=base_model.input, outputs=base_model.get_layer("avg_pool").output)

        @staticmethod
        def preprocess_image_array(image_array):
            return preprocess_input(image_array)


.. _VGG16: https://keras.io/api/applications/vgg/#vgg16-function
.. _InceptionV3: https://keras.io/api/applications/inceptionv3/
