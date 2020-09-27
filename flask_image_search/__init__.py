import logging
import os
import threading

import numpy as np
from PIL import Image
from sqlalchemy import event, column
from sqlalchemy.orm import query_expression, with_expression
from sqlalchemy.sql.expression import case

__author__ = """Hanan Fokkens"""
__email__ = 'hananfokkens@gmail.com'
__version__ = '0.1.0'

# get the logger
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s flask image search: %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.INFO)


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
                    # copy accross
                    self.set(key, file[key])

    def __delitem__(self, name):
        super().__delitem__(name)
        self.save()

    def __setitem__(self, name, value):
        super().__setitem__(name, value)
        self.save()

    def set(self, name, value):
        """Set a item without saving."""
        super().__setitem__(name, value)

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
        :param tensorflow: Should tensorflow be used, tensorflow can be dissabled so debug mode can be used easier,
            defaults to True
        :type tensorflow: bool
        :raises Exception: Exception may be raised if Flask-SQLAlchemy is not initialized.
        """
        app.config.setdefault("IMAGE_SEARCH_PATH_PREFIX", "image_search/")

        self.root = app.root_path
        self.path_prefix = app.config["IMAGE_SEARCH_PATH_PREFIX"]

        sqlalchemy = app.extensions.get('sqlalchemy')

        if sqlalchemy is None:
            raise Exception("You need to initialize Flask-SQLAlchemy before Flask-Image-Search.")

        self.db = sqlalchemy.db

        if tensorflow:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # set tensorflow debug level to only show errors
            from .feature_extractor import FeatureExtractor
        else:
            from .feature_extractor_debug import FeatureExtractor
        self.feature_extractor = FeatureExtractor()
        self.models = {}

    def register(self, id='id', url='url', ignore='ignore', foreign=None):
        """This decorator is used to register Flask-SQLAlchemy Model with  the image search.
        After a model is registered it can then indexed and searched.

        :param id: This is the name of the primary_key column. Defaults to 'id'.
        :type id: str
        :param url: This is the name of the column containing the image url. Defaults to 'url'.
        :type url: str
        :param ignore: This is the name of the column used to decide if an image should be ignored.
            defaults to 'ignore'.
        :type ignore: str
        :param foreign: A list of foreign_keys to store.
            This is used so image search works on Models related to the Model with images. Defaults to None
        :type foreign: list[str] or None
        """
        if foreign is None:
            foreign = []

        def inner(model):
            # work out the file path
            file_path = os.path.join(self.root, self.path_prefix + model.__tablename__ + '.npz')

            # Store the information about this model in the models dict.
            self.models[model.__tablename__] = dict(
                features=sdict(file_path),
                id=id,
                url=url,
                ignore=ignore if hasattr(model, ignore) else False,
                foreign=foreign
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

            model.query_class.image_search = lambda self_, *args, **kwargs: self.query_search(*args, **kwargs)(self_)

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
        return self.models[model]['features']

    def image_id(self, entry):
        """Helper function to genarate the id used to identify an image

        :param entry: The model instance to get the id from
        :type entry: flask_sqlalchemy.Model
        """
        data = self.models[entry.__tablename__]  # get the data for this entry

        # create list of the parts for the image id
        image_id_parts = [str(getattr(entry, data['id']))]

        # add all the foreign keys to the image_id_parts list
        for fk in data['foreign']:
            image_id_parts.append(str(getattr(entry, fk)))

        image_id = "_".join(image_id_parts)  # join all the parts with an underscore
        return image_id

    def index(self, entry, replace=False):
        """This method indexes an entry of a registered Model.

        :param entry: The Model instance to be indexed.
        :type entry: flask_sqlalchemy.Model
        :param replace: Set to True to replace an existing index of this image, defaults to False
        :type replace: bool
        """
        data = self.models[entry.__tablename__]  # get the data for this entry

        if data['ignore'] and getattr(entry, data['ignore']):
            # if this model is ignored return false
            return False

        image_path = getattr(entry, data['url'])  # get the image url

        image_id = self.image_id(entry)

        if not replace and image_id in data['features']:
            # if the image isn't allowed to be reindexed and it already is indexed skip it
            return True

        # open the image
        image = Image.open(os.path.join(self.root, image_path.strip('/')))
        image_feature = self.feature_extractor.extract(image)  # extract the features
        data['features'][image_id] = image_feature  # save the features to the features dict
        return True

    def index_model(self, model, replace=False):
        """Index all the images in a model.

        :param model: The model containing the images to be indexed.
        :type model: flask_sqlalchemy.Model
        :param replace: Set to True to replace an existing index of this image, defaults to False
        :type replace: bool
        """
        entries = self.db.session.query(model).all()

        total = 0
        indexed = 0

        for entry in entries:
            # index each entry
            total += 1
            if self.index(entry, replace):
                indexed += 1

        logger.info(f"Indexed {indexed} of {total} images for the model {model.__tablename__}")

    def delete_index(self, entry):
        """Delete an index

        :param entry: The entry in the database to be removed from the index.
        :type entry: flask_sqlalchemy.Model
        """
        data = self.models[entry.__tablename__]  # get the data related to this entry

        # get the image id
        image_id = self.image_id(entry)
        try:
            del data['features'][image_id]
        except KeyError:
            raise KeyError("That Image is not indexed.")

    def search(self, model, image, limit=None):
        """This searches the indexed data with an image and returns a tuple of id strings.

        :param model: This is the model to be search.
            Either a SQLAlchemy model can be passed or a string of the tablename.
        :type model: str or flask_sqlalchemy.Model
        :param image: The search image
        :type image: PIL.Image.Image or str
        :param limit: The number of results to be returned, defaults to None
        :type limit: int or None
        :return: This returns a tuple of tuples containing the id and the distance from the search image.
        :rtype: tuple[tuple[str, int]]
        """
        if type(image) is str:
            # if the image is a string open it as a PIL image
            image = Image.open(os.path.join(self.root, image))

        if type(model) is not str:
            model = model.__tablename__

        search_features = self.feature_extractor.extract(image)  # extract the features form the search image.

        ids, features = zip(*self.models[model]['features'].items())

        # get the distance between all the indexed images and the search image.
        distances = np.linalg.norm(features - search_features, axis=1)
        # argsort [1, 3, 2] --> [0, 2, 1]
        distances_id_sorted = np.argsort(distances)  # get the order to apply to sort the images

        if limit is not None:
            distances_id_sorted = distances_id_sorted[:limit]

        # return a list of the ids and distances from the search image
        return tuple((ids[id], distances[id]) for id in distances_id_sorted)

    def get_model_from_query(self, query):
        """This is a helper function that tries to get a model from a query.

        :param query: The query to get th model from.
        :type query: flask_sqlalchemy.BaseQuery
        """
        pass

    def query_search(self, image, limit=20, model_=None, hard=True):
        """This is used to search filter a model using an image.
        query_search calls `search` and adds the results to the query,
        by adding a case statement under the column `distance`.
        This case statement is used in the order added to the query,
        and if `hard` is set to True then is also added as a filter.
        This distance can be accessed in your query results as `distance` column on your model.

        :param image: the search image. this can be a path string or a PIL image.
        :type image: PIL.Image.Image or str
        :param limit: the number of results, defaults to 20
        :type limit: int
        :param model_: The model that has been registered.
            If ths is set to None this will try and use the query to find the Model, defaults to None
        :type model_: flask_sqlalchemy.Model
        :param hard: If this is set to True the query will be trimmed so that it only the limit.
            Setting this to false will sort `limit` results and then return all other images without a set order,
            defaults to True
        :type hard: bool
        :return: returns a function
        :rtype: function
        """
        if type(image) is str:
            image = Image.open(image)

        def inner(query):
            model = model_
            # if the image is none just do nothing to the query.
            if image is None:
                return query

            # get the flask sqlalchemy model form column descriptions
            if model is None:
                # construct a set of possible models
                models = set(
                    col['type'] for col in query.column_descriptions if hasattr(col['type'], '__tablename__')
                )
                if len(models) > 1:
                    raise Exception("Can't work out model from query, please set model to the correct model.")
                else:
                    model = models.pop()

            results = self.search(model, image, limit)

            data = self.models[model.__tablename__]

            # get the id column
            model_id_column = getattr(model, data['id'])

            ids = []
            case_comparisons = []

            for id, distance in results:
                ids.append(id.split("_")[0])  # add the id
                case_comparisons.append(((
                    model_id_column == id.split("_")[0]),
                    float(distance)
                ))
            case_statement = case(case_comparisons, else_=None).label("distance")

            query = query.options(with_expression(model.distance, case_statement)).order_by("distance")
            if hard:
                query = query.filter(column("distance"))

            return query

        return inner
