import logging
import os
import threading
from types import SimpleNamespace
from flask_image_search.__about__ import __author__, __author_email__, __version__, __license__

import numpy as np
import zarr
from PIL import Image
from sqlalchemy import case
from sqlalchemy import column as sa_column
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
        from tensorflow import keras
        base_model = keras.applications.vgg16.VGG16(weights="imagenet")
        return keras.Model(inputs=base_model.input, outputs=base_model.get_layer("fc1").output)

    @staticmethod
    def preprocess_image_array(image_array):
        from tensorflow.keras.applications.vgg16 import preprocess_input
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
            # Store the information about this model in the models dict.
            self.models[model.__tablename__] = SimpleNamespace(
                features=self.storage.require_group(model.__tablename__ + '_features'),
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
            logger.info(f"Loaded {len(data.features)} image features for '{model.__tablename__}'")

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
        return self.storage.require_group(model + '_features')

    def feature_extract(self, image):
        """This is a helper function that takes an image processes it and returns the features.

        :param image: The image to get the features from.
        :type image: PIL.Image.Image
        """
        from keras.preprocessing.image import img_to_array

        if self.keras_model:
            image_size = self.keras_model.inputs[0].shape[1:3]
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

        image_id = str(getattr(entry, data.id))

        if not replace and image_id in data.features:
            # if the image isn't allowed to be reindexed and it already is indexed skip it
            return True

        # open image
        image = Image.open(os.path.join(self.root, image_path))

        features = self.feature_extract(image)
        # save the features in a dataset named with the image_id
        data.features.require_dataset(image_id, features.shape, features.dtype, data=features)
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
            entries = self.db.session.query(model).all()

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
        image_id = str(getattr(entry, data.id))
        try:
            del data.features[image_id]
        except KeyError:
            raise KeyError("That Image is not indexed.")

    def search(self, model, image):
        """This searches the indexed data with an image and returns a tuple of id strings.

        :param model: This is the model to be search.
            Either a SQLAlchemy model can be passed or a string of the tablename.
        :type model: str or flask_sqlalchemy.Model
        :param image: The search image
        :type image: PIL.Image.Image or str
        :return: This returns a tuple of tuples containing the id and the distance from the search image.
        :rtype: tuple[tuple[str, int]]
        """
        if type(image) is str:
            # if the image is a string open it as a PIL image
            image = Image.open(os.path.join(self.root, image))

        if type(model) is not str:
            model = model.__tablename__

        search_features = self.feature_extract(image)  # extract the features form the search image.

        if len(self.models[model].features) > 0:
            ids, features = zip(*self.models[model].features.items())
        else:
            raise Exception("You must index some images before you can search")

        # get the distance between all the indexed images and the search image.
        distances = np.linalg.norm(features - search_features, axis=1)
        # argsort [1, 3, 2] --> [0, 2, 1]
        distances_id_sorted = np.argsort(distances)  # get the order to apply to sort the images

        # return a list of the ids and distances from the search image
        return tuple((ids[id], distances[id]) for id in distances_id_sorted)

    def case(self, image, model, column=None):
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

        results = self.search(model, image)  # get the ids and distances

        whens = []

        # construct the whens for the case statement
        for id, distance in results:
            whens.append((
                # literal columns insted of bind parameters
                (column == literal_column(id.split("_")[0])), literal_column(str(distance))
            ))

        return case(whens, else_=None)
