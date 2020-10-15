import os

from flask import Flask
from flask_sqlalchemy import SQLAlchemy

from flask_image_search import ImageSearch

app = Flask(__name__)
app.config.update({
    "SQLALCHEMY_DATABASE_URI": f"sqlite:///{app.root_path}/../resources/test.db",
    "SQLALCHEMY_TRACK_MODIFICATIONS": False,
})

db = SQLAlchemy(app)
image_search = ImageSearch(app)


class Radio(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String, nullable=False)

    images = db.relationship('Image')


@image_search.register()
class Image(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    path = db.Column(db.String, nullable=False)
    radio_id = db.Column(db.Integer, db.ForeignKey("radio.id"), nullable=False)


image_search.index_model(Image)  # index the model so it can be searched

# search with an image using query_search
images = Image.query.with_transformation(image_search.query_search("test.jpg")).all()
print(images)

# search using query.image_search
images = Image.query.image_search("../resources/test.jpg").all()
print(images)

# join search using query.image_search
query = Radio.query.join(Image).options(db.contains_eager(Radio.images))
query = query.image_search("../resources/test.jpg", join=True)
radios = query.all()
print(radios)
