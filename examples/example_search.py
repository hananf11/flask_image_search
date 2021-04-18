from flask import Flask
from flask_sqlalchemy import SQLAlchemy

from flask_image_search import ImageSearch


app = Flask(__name__)
app.config.update({
    "SQLALCHEMY_DATABASE_URI": f"sqlite:///{app.root_path}/test.db",
    "SQLALCHEMY_TRACK_MODIFICATIONS": False,
})

db = SQLAlchemy(app)
image_search = ImageSearch(app)


class Radio(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String, nullable=False)

    images = db.relationship("Image", backref="radio", primaryjoin="Radio.id == Image.radio_id")

    def __repr__(self):
        return f"<Radio {self.id} {self.name}>"


@image_search.register()
class Image(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    path = db.Column(db.String, nullable=False)
    radio_id = db.Column(db.Integer, db.ForeignKey("radio.id"), nullable=False)

    def __repr__(self):
        return f"<Image {self.id} {self.radio.name}>"


image_search.index_model(Image, threaded=False)

images = Image.query.order_by(image_search.case("./test.jpg", Image)).all()

print(images)

radios = Radio.query.join(Radio.images).options(db.joinedload(Radio.images)).order_by(
    image_search.case("./test.jpg", Image)).all()

print(radios)
