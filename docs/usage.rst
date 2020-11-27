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


Searching
---------

With Flask-Image-Search you can search on registred models and models that have a relationship to a registered model.

Basic searching
^^^^^^^^^^^^^^^

To do a search :meth:`ImageSearch.query_search()`, query_search returns a function that takes a query and returns a :class:`Query <sqlalchemy.orm.query.Query>`.
query_search is designed to be used with :meth:`Query.with_transformation() <sqlalchemy.orm.query.Query.with_transformation()>` like so::

    images = Image.query.with_transformation(image_search.query_search('my_image.jpg')).all()

The easiest way to do a search is to use the :meth:`Query.image_search` method that is added when :class:`ImageSearch` is initialized,
this is an alias to :meth:`ImageSearch.query_search()` so it takes all the same parameters::

    images = Image.query.image_search('my_image.jpg').all()


Join searching
^^^^^^^^^^^^^^

.. warning::
    This only works with a one to many relationship where the images are the many.

It is possible to search a Model that does not contain images but is related to one that does and has been indexed, using a join::

    class Animals(db.Model):
        id = db.Column(db.Integer, primary_key=True)
        name = db.Column(db.Text)

        images = db.relationship("Image")

    @image_search.register()
    class Image(db.Model):
        ...
        animal_id = db.Column(db.Integer, db.ForeignKey("animal.id"))

    animals = Animals.query.join(Animals.images).options(db.contains_eager(Animals.images)) \
        .image_search('my_image.jpg', join=True).all()

Short hand::

    animals = Animals.query.image_search('my_image.jpg', join=Animals.images).all()

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
