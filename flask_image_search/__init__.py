import logging
import os
import threading

import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model as KerasModel
from keras.preprocessing import image as keras_image
from PIL import Image
from sqlalchemy import event, literal_column, nullslast
from sqlalchemy.orm import contains_eager, query_expression, with_expression
from sqlalchemy.sql.expression import case
from sqlalchemy_utils import get_query_entities, get_type

__author__ = """Hanan Fokkens"""
__email__ = "hananfokkens@gmail.com"
__version__ = "0.5.0"

# get the logger
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s flask image search: %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # set tensorflow debug level to only show errors


class sdict(dict):
    """Dictionary that uses numpy save to save itself when it is updated."""

    def __init__(self, file, *args, **kwargs):
        self.file = file
        self.save_countdown = None
        self.call_count = 0  # how many times has this function been called
        super().__init__(*args, **kwargs)

    def save(self):
        if self.save_countdown is not None:
            self.save_countdown.cancel()
            self.call_count += 1

        def save():
            self.call_count = 0
            if not os.path.exists(os.path.dirname(self.file)):
                os.mkdir(os.path.dirname(self.file))
            np.savez_compressed(self.file, **self)

        self.save_countdown = threading.Timer(1, save)
        self.save_countdown.start()
        if self.call_count >= 100:
            self.save_countdown.join()

    def load(self):
        # check the file exists to ovid errors
        if os.path.exists(self.file):
            with np.load(self.file) as file:
                for key in file.files:
                    # copy accross without triggering save
                    super().__setitem__(key, file[key])

    def __delitem__(self, name):
        super().__delitem__(name)
        self.save()

    def __setitem__(self, name, value):
        super().__setitem__(name, value)
        self.save()

    def clear(self):
        super().clear()
        self.save()

    def pop(self):
        result = super().pop()
        self.save()
        return result

    def popitem(self):
        result = super().popitem()
        self.save()
        return result

    def setdefault(self, key, default):
        result = super().setdefault(key, default)
        self.save()
        return result

    def update(self, __m, **kwargs):
        super().update(__m, **kwargs)
        self.save()


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
        app.config.setdefault("IMAGE_SEARCH_PATH_PREFIX", "image_search/")

        # get the path_prefix and the app root
        self.path_prefix = app.config["IMAGE_SEARCH_PATH_PREFIX"]
        self.root = app.root_path

        # get db from sqlalchemy
        sqlalchemy = app.extensions.get("sqlalchemy")
        if sqlalchemy is None:
            raise Exception("You need to initialize Flask-SQLAlchemy before Flask-Image-Search.")
        self.db = sqlalchemy.db

        # add alias to query_search in db.Query
        self.db.Query.image_search = lambda self_, *args, **kwargs: self.query_search(*args, **kwargs)(self_)

        if tensorflow:
            self.keras_model = self.create_keras_model()
        else:
            self.keras_model = None
        self.models = {}

    @staticmethod
    def create_keras_model():
        """This functions exists so that `tensorflow=False` works with a custom model."""
        base_model = VGG16(weights="imagenet")
        return KerasModel(inputs=base_model.input, outputs=base_model.get_layer("fc1").output)

    @staticmethod
    def preprocess_image_array(image_array):
        return preprocess_input(image_array)

    def register(self, id="id", path="path", ignore="ignore"):
        """This decorator is used to register Flask-SQLAlchemy Model with  the image search.
        After a model is registered it can then indexed and searched.

        :param id: This is the name of the primary_key column. Defaults to "id".
        :type id: str
        :param path: This is the name of the column containing the image path. Defaults to "path".
        :type path: str
        :param ignore: This is the name of the column used to decide if an image should be ignored.
            defaults to "ignore".
        :type ignore: str
        """
        def inner(model):
            # work out the file path
            file_path = os.path.join(self.root, self.path_prefix + model.__tablename__ + ".npz")

            # Store the information about this model in the models dict.
            self.models[model.__tablename__] = dict(
                features=sdict(file_path),
                id=id,
                path=path,
                ignore=ignore if hasattr(model, ignore) else False,
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

            # load image features for this model
            data["features"].load()
            logger.info(f"Loaded in {len(data['features'])} image indexes for the model {model.__tablename__}")

            model.distance = query_expression()

            return model

        return inner

    def features(self, model):
        """Returns features of a model.

        :param model: The model that you want to get the features
        :type model: flask_sqlalchemy.Model or str
        :return: Returns the features of a model.
        :rtype: dict
        """
        if type(model) is not str:
            model = model.__tablename__
        return self.models[model]["features"]

    def feature_extract(self, image):
        """This is a helper function that takes an image processes it and returns the features.

        :param image: The image to get the features from.
        :type image: PIL.Image.Image
        """
        if self.keras_model:
            image_size = self.keras_model.inputs[0].shape[1:3]
            image = image.resize(image_size).convert("RGB")  # resize the image and convert to RGB
            image_array = keras_image.img_to_array(image)  # turn image into np array
            image_array = np.expand_dims(image_array, axis=0)  # expand the shape of array
            input_array = self.preprocess_image_array(image_array)

            feature = self.keras_model.predict(input_array)[0]  # process
            return feature / np.linalg.norm(feature)  # normalize the features
        else:
            return np.random.rand(4096)

    def index(self, entry, replace=False):
        """This method indexes an entry of a registered Model.

        :param entry: The Model instance to be indexed.
        :type entry: flask_sqlalchemy.Model
        :param replace: Set to True to replace an existing index of this image, defaults to False
        :type replace: bool
        """
        data = self.models[entry.__tablename__]  # get the data for this entry

        if data["ignore"] and getattr(entry, data["ignore"]):
            # if this model is ignored return false
            return False

        image_path = getattr(entry, data["path"])  # get the image path

        image_id = str(getattr(entry, data["id"]))

        if not replace and image_id in data["features"]:
            # if the image isn't allowed to be reindexed and it already is indexed skip it
            return True

        # open image
        image = Image.open(os.path.join(self.root, image_path))

        data["features"][image_id] = self.feature_extract(image)  # save the features to the features dict
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
        image_id = str(getattr(entry, data["id"]))
        try:
            del data["features"][image_id]
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

        ids, features = zip(*self.models[model]["features"].items())

        # get the distance between all the indexed images and the search image.
        distances = np.linalg.norm(features - search_features, axis=1)
        # argsort [1, 3, 2] --> [0, 2, 1]
        distances_id_sorted = np.argsort(distances)  # get the order to apply to sort the images

        # return a list of the ids and distances from the search image
        return tuple((ids[id], distances[id]) for id in distances_id_sorted)

    def query_search(self, image, image_model=None, query_model=None, join=False):
        """This is used to search filter a model using an image.
        query_search calls `search` and adds the results to the query,
        by adding a case statement under the column `distance`.
        This case statement is used in the order added to the query.

        :param image: the search image. this can be a path string or a PIL image.
        :type image: PIL.Image.Image or str
        :param image_model: The model that has been registered, the one containing the image.
            If ths is set to None the query will be used to find this value, defaults to None
        :type image_model: flask_sqlalchemy.Model
        :param query_model: The model is being queried, this is only used when join is True.
            When this is set to None the query will be used to find this value.
        :type query_model: flask_sqlalchemy.Model
        :param join: Set this to join mode.
        :type join: bool or flask_sqlalchemy.Model
        :return: returns a function
        :rtype: function
        """
        def inner(query):
            image_model_ = image_model
            query_model_ = query_model

            if join is not True and join is not False:
                query = query.join(join).options(contains_eager(join))

            # if the image is none just do nothing to the query.
            if image is None:
                return query

            entities = get_query_entities(query)

            if query_model_ is None and join:
                query_model_ = entities[0]  # in join mode query_model_ is the first entity

            # get the flask sqlalchemy model form column descriptions
            if image_model_ is None:
                if join:
                    # in join mode the image_model_ is in a mapper in the second entity
                    image_model_ = entities[1].class_
                else:
                    image_model_ = entities[0]  # in the nomral mode the image_model_ is the first entity

            data = self.models[image_model_.__tablename__]

            expression = image_model_.distance

            results = self.search(image_model_, image)  # get the ids and distances

            # get the id column so it can be used in the case statment
            id_column = getattr(image_model_, data["id"])

            if join:

                # update the exspression column statment
                for key, value in query_model_.__mapper__._props.items():
                    if get_type(value) is image_model_:
                        expression = f"{key}.distance"

            whens = []

            # construct the whens for the case stament
            for id, distance in results:
                whens.append((
                    # literal columns insted of bind parameters
                    (id_column == literal_column(id.split("_")[0])),
                    literal_column(str(distance))
                ))

            case_statement = case(whens, else_=None).label("distance")

            query = query.options(with_expression(expression, case_statement))
            query = query.order_by(nullslast("distance"))

            return query

        return inner
