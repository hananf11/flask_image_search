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


Registering Models
------------------

On any :class:`flask_sqlalchemy.Model` that you want to index you will need to register.
Registering a Model can be done using the :meth:`ImageSearch.register()` decorator.

To register a Model::

    @image_search.register()
    class Image(db.Model):
        id = db.Column(db.Integer, primary_key=True)
        url = db.Column(db.Text)

If you want to search on a Model that is related to your image model,
you must specify the foreign key columns when registering::

    @image_search.register(foreign=['animal_id'])
    class Image(db.Model):
        id = db.Column(db.Integer, primary_key=True)
        url = db.Column(db.Text)
        animal_id = db.Column(db.ForeignKey('animals.id'))

        animal = db.relationship('Animals', primaryjoin='Image.animal_id == Animals.id', backref="images")

    class Animals(db.Model):
        id = db.Column(db.Integer, primary_key=True)
        name = db.Column(db.Text)

Indexing
--------

Once a Model has been registered the images must be indexed. Images that are not indexed will not be searched.

To index one image of a registered Model::

    image = Image.query.first()  # get an image however you like

    image_search.index(image)

To index all images of a registered Model::

    image_search.index_model(Image)

Once Images are indexed changes on the database will be reflected on the index.
The indexes will be updated when an image is dropped, updated or a new image is added.

It is possible to manually delete an image from the index::

    image = Image.query.first()

    image_search.delete(Image)


