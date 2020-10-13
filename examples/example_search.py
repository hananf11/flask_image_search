import os

from flask import Flask
from flask_sqlalchemy import SQLAlchemy

from flask_image_search import ImageSearch

app = Flask(__name__)
app.config.update({
    "SQLALCHEMY_DATABASE_URI": f"sqlite:///{app.root_path}/../resources/test.db",
    "SQLALCHEMY_TRACK_MODIFICATIONS": False,
    "IMAGE_SEARCH_PATH_PREFIX": "../resources/image_search_example_search/",
})

db = SQLAlchemy(app)
image_search = ImageSearch(app)


class Model(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String, nullable=False)

    images = db.relationship('Image')


@image_search.register()
class Image(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    path_ = db.Column('path', db.String, nullable=False)
    model_id = db.Column(db.Integer, db.ForeignKey("model.id"), nullable=False)

    @property
    def path(self):
        return os.path.join('../resources/', self.path_)


image_search.index_model(Image)
print(Image.query.with_transformation(image_search.query_search("../resources/test.jpg")).all())
print(Image.query.image_search("../resources/test.jpg"))
