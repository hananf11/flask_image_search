import logging
import math
import os
import threading
from types import SimpleNamespace
from flask_image_search.__about__ import __author__, __author_email__, __version__, __license__

import numpy as np
import zarr
from PIL import Image
from sqlalchemy import case
from sqlalchemy import column as sa_column
from sqlalchemy.orm import lazyload
from sqlalchemy import event, literal_column

# get the logger
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s flask image search: %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # set tensorflow debug level to only show errors

__all__ = (
    'ImageSearch',
    __version__,
    __author__,
    __author_email__,
    __license__
)


class ImageSearch(object):
    """This is a class used to add Image searching functionality to your flask app,
    that uses Flask-SQLAlchemy.

    There are two ways to use Flask-Image-Search.

    .. code-block:: python

        app = Flask(__name__)
        db = SQLAlchemy(app)
        image_search = ImageSearch(app, db)

    or

    .. code-block:: python

        db = SQLAlchemy()
        image_search = ImageSearch()

        def create_app():
            app = Flask(__name__)
            db.init_app(app)
            image_search.init_app(app, db)
            return app

    """

    __slots__ = ['root', 'storage', 'db', 'keras_model', 'models', 'app']

    def __init__(self, app=None, **kwargs):
        self.app = app
        if app is not None:
            self.init_app(app, **kwargs)

    def init_app(self, app, tensorflow=True):
        """Initialize Flask-Image-Search. This function is called by __init__ if an app is passed.

        :param app: The Flask app for this project.
        :type app: flask.Flask
        :param tensorflow: Should tensorflow be used,
            tensorflow can be dissabled to make working on other parts of your app easier
            defaults to True
        :type tensorflow: bool
        :raises Exception: Exception may be raised if Flask-SQLAlchemy is not initialized.
        """
        app.config.setdefault("IMAGE_SEARCH_PATH", "image_search")

        # get the path_prefix and the app root
        self.root = app.root_path
        path = os.path.join(self.root, app.config["IMAGE_SEARCH_PATH"])
        self.storage = zarr.open(path, mode='a')

        # get db from sqlalchemy
        sqlalchemy = app.extensions.get("sqlalchemy")
        if sqlalchemy is None:
            raise Exception("You need to initialize Flask-SQLAlchemy before Flask-Image-Search.")
        self.db = sqlalchemy.db

        if tensorflow:
            self.keras_model = self.create_keras_model()
        else:
            self.keras_model = None
        self.models = {}

    @staticmethod
    def create_keras_model():
        """This functions exists so that `tensorflow=False` works with a custom model."""
        import keras
        base_model = keras.applications.vgg16.VGG16(weights="imagenet")
        return keras.Model(inputs=base_model.input, outputs=base_model.get_layer("fc1").output)

    @staticmethod
    def preprocess_image_array(image_array):
        from keras.applications.vgg16 import preprocess_input
        return preprocess_input(image_array)

    def register(self, id="id", path="path", ignore="ignore"):
        """This decorator is used to register Flask-SQLAlchemy Model with  the image search.
        After a model is registered it can then indexed and searched.

        :param id: This is the name of the primary_key column. Defaults to "id".
        :type id: str
        :param path: This is the name of the column containing the image path. Defaults to "path".
        :type path: str
        :param ignore: This is the name of the column used to decide if an image should be ignored.
            defaults to "ignore" if no column is found there is no ignore column.
        :type ignore: str
        """
        def inner(model):
            from keras import backend
            size = backend.int_shape(self.keras_model.outputs[0])[1]
            # backend.int_shape(self.keras_model.outputs[0])[1:3]
            if model.__tablename__ + '_features' not in self.storage:
                self.storage.create_dataset(
                    model.__tablename__ + '_features',
                    shape=(5000, size),
                    chunks=(5000, size),
                    dtype=np.float32
                )
            # Store the information about this model in the models dict.
            self.models[model.__tablename__] = SimpleNamespace(
                features=self.storage[model.__tablename__ + '_features'],
                id=id,
                path=path,
                ignore=ignore if ignore and hasattr(model, ignore) else False,
            )

            data = self.models[model.__tablename__]

            # add events so that the changes on the database are reflected in indexed images.
            @event.listens_for(model, "after_delete")
            def image_search_model_deleted(mapper, connection, target):
                """A model was deleted."""
                self.delete_index(target)

            @event.listens_for(model, "after_insert")
            def image_search_model_insert(mapper, connection, target):
                """A new model was inserted."""
                self.index(target)

            @event.listens_for(model, "after_update")
            def image_search_model_updated(mapper, connection, target):
                """An existing model was updated."""
                self.index(target, True)

            # log how many images were loaded
            logger.info(
                f"Loaded {int(np.sum(np.any(data.features[:] != 0, axis=1)))} image features for '{model.__tablename__}'")

            return model

        return inner

    def features(self, model):
        """Returns features of a model.

        :param model: The model that you want to get the features
        :type model: flask_sqlalchemy.Model or str
        :return: The group containing the datasets for each image
        :rtype: zarr.hierarchy.Group
        """
        if type(model) is not str:
            model = model.__tablename__
        return self.storage.require_dataset(model + '_features')

    def feature_extract(self, image):
        """This is a helper function that takes an image processes it and returns the features.

        :param image: The image to get the features from.
        :type image: PIL.Image.Image
        """
        from keras.preprocessing.image import img_to_array
        from keras import backend

        if self.keras_model:
            image_size = backend.int_shape(self.keras_model.inputs[0])[1:3]
            image = image.resize(image_size).convert("RGB")  # resize the image and convert to RGB
            image_array = img_to_array(image)  # turn image into np array
            image_array = np.expand_dims(image_array, axis=0)  # expand the shape of array
            input_array = self.preprocess_image_array(image_array)

            feature = self.keras_model.predict(input_array)[0]  # process
            return feature / np.linalg.norm(feature)  # normalize the features
        else:
            return np.random.rand(4096)

    def index(self, entry, replace=False):
        """Indexes an entry of a registered Model.

        :param entry: The Model instance to be indexed.
        :type entry: flask_sqlalchemy.Model
        :param replace: Set to True to replace an existing index of this image, defaults to False
        :type replace: bool
        """
        data = self.models[entry.__tablename__]  # get the data for this entry

        if data.ignore and getattr(entry, data.ignore):
            # if this model is ignored return false
            return False

        image_path = getattr(entry, data.path)  # get the image path
        image_id = getattr(entry, data.id)

        if image_id >= data.features.shape[0] - 1:
            data.features.resize((int(math.ceil(image_id / 1000) * 1000), data.features.shape[1]))

        if not replace and (data.features[image_id] != 0).any():
            # if the image isn't allowed to be reindexed and it already is indexed skip it
            return True

        # open image
        image = Image.open(os.path.join(self.root, image_path))

        features = self.feature_extract(image)
        # save the features in a dataset named with the image_id
        data.features[image_id] = features
        return True

    def index_model(self, model, replace=False, threaded=True):
        """Index all the images in a model.

        :param model: The model containing the images to be indexed.
        :type model: flask_sqlalchemy.Model
        :param replace: Set to True to replace an existing index of this image, defaults to False
        :type replace: bool
        :param threaded: Set this to true to make index_model non blocking.
        :type threaded: bool
        """
        def thread_content():
            entries = self.db.session.query(model).options(lazyload('*')).all()

            total = 0
            indexed = 0

            for entry in entries:
                # index each entry
                total += 1
                if self.index(entry, replace):
                    indexed += 1

            logger.info(f"Indexed {indexed} of {total} images for the model {model.__tablename__}")

        if threaded:
            threading.Thread(target=thread_content).start()
        else:
            thread_content()

    def delete_index(self, entry):
        """Delete an index

        :param entry: The entry in the database to be removed from the index.
        :type entry: flask_sqlalchemy.Model
        """
        data = self.models[entry.__tablename__]  # get the data related to this entry

        # get the image id
        image_id = getattr(entry, data.id)
        try:
            data.features[image_id] = np.zeros(data.features[image_id].shape)
        except KeyError:
            raise KeyError("That Image is not indexed.")

    def search(self, model, image, sorted=True, limit=None):
        """This searches the indexed data with an image and returns a tuple of id strings.

        :param model: This is the model to be search.
            Either a SQLAlchemy model can be passed or a string of the tablename.
        :type model: str or flask_sqlalchemy.Model
        :param image: The search image
        :type image: PIL.Image.Image or str
        :param sorted: Should the results be sorted
        :type sorted: bool
        :param limit: Limit the number of results
        :type limit: None or int
        :return: This returns a tuple of tuples containing the id and the distance from the search image.
        :rtype: tuple[tuple[str, int]]
        """
        if type(image) is str:
            # if the image is a string open it as a PIL image
            image = Image.open(os.path.join(self.root, image))

        if type(model) is not str:
            model = model.__tablename__

        search_features = self.feature_extract(image)  # extract the features form the search image.

        # genorate zero for the feature size
        zeros = np.zeros((1, search_features.shape[0]))
        # get the distance between all the indexed images and the search image.
        # concatonate between the zeros and the features so that the first value is an un-indexed reference
        distances = np.linalg.norm(np.concatenate([zeros, self.models[model].features]) - search_features, axis=1)
        if sorted:
            distances_id_sorted = np.argsort(distances)
        else:
            distances_id_sorted = range(len(distances))

        # return a list of the ids and distances from the search image
        # reduce the id by one to account for the zeros
        results = tuple((id - 1, distances[id]) for id in distances_id_sorted if distances[id] != distances[0])
        if limit is not None:
            results = results[:limit]
        return results

    def case(self, image, model, column=None, limit=None):
        """Creates a case statement that contains the distances to the query image matching up to ids.

        :param image: The query image
        :type image: PIL.Image.Image or str
        :param model: The model containing the indexed images.
        :type model: flask_sqlalchemy.Model
        :param column: The column that the case statement relates to (Primary key column of the indexed Model)
        :type column: str or sqlalchemy.schema.Column
        """
        if column is None:
            column = getattr(model, self.models[model.__tablename__].id)
        if type(column) is str:
            column = sa_column(column)

        results = self.search(model, image, sorted=limit is not None, limit=limit)  # get the ids and distances

        whens = []

        # construct the whens for the case statement
        for id, distance in results:
            if limit and len(whens) >= limit:
                break
            whens.append((
                # literal columns insted of bind parameters
                (column == literal_column(str(id))), literal_column(str(distance))
            ))

        if whens:
            return case(whens, else_=literal_column('9999'))
        else:
            return None
