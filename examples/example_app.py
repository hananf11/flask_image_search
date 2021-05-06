from flask import Flask, render_template_string, request, redirect, flash
from flask_sqlalchemy import SQLAlchemy
from flask_image_search import ImageSearch
from PIL import Image as PILImage

app = Flask(__name__)
app.config.update({
    "SQLALCHEMY_DATABASE_URI": f"sqlite:///{app.root_path}/test.db",
    "SQLALCHEMY_TRACK_MODIFICATIONS": False,
    "SECRET_KEY": "BAD SECRET KEY"
})


db = SQLAlchemy(app)
image_search = ImageSearch(app)


# Models
class Radio(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String, nullable=False)

    images = db.relationship("Image", backref="radio")


@image_search.register()
class Image(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    path = db.Column(db.String, nullable=False)
    radio_id = db.Column(db.Integer, db.ForeignKey("radio.id"), nullable=False)

    distance = db.query_expression()


image_search.index_model(Image)


@app.route("/", methods=["GET", "POST"])
def home():
    images = []
    radios = []
    if request.method == "POST":
        f = request.files.get("image")
        if not f:
            flash("Failed to get image")
            return redirect("/")
        image = PILImage.open(f)
        case = image_search.case(image, Image).label('d')
        images = Image.query.options(db.with_expression(Image.distance, case)).order_by('d')

        radios = Radio.query.join(Radio.images).options(
            db.contains_eager(Radio.images).with_expression(Image.distance, case)
        ).order_by('d')

    return render_template_string("""
    {{ get_flashed_messages()[0] }}
    <form method="POST" enctype="multipart/form-data" >
        <input type="file" name="image" accept="image/x-png,image/gif,image/jpeg" />
        <button type="submit">Upload</button>
    </form>
    <table>
        <tr>
            <th>Images:</th>
            <th>Radios:</th>
        </tr>
        <tr>
            <td valign=top>
                <ol>
                {% for image in images %}
                    <li>
                        Distance: {{ image.distance }}<br>
                        Img: <img src="/{{ image.path }}" height=100>
                    </li>
                {% endfor %}
                </ol>
            </td>
            <td valign=top>
                <ol>
                {% for radio in radios %}
                    <li>
                        Name: {{ radio.name }}
                        <ol>
                        {% for image in radio.images %}
                            <li>
                                Name: {{ image.radio.name }}
                                Distance: {{ image.distance }}<br>
                                Img: <img src="/{{ image.path }}" height=100>
                            </li>
                        {% endfor %}
                        </ol>
                    </li>
                {% endfor %}
                </ol>
            </td>
        </tr>
    </table>
    """, images=images, radios=radios)


if __name__ == "__main__":
    app.run(debug=True)
