import os
import sys

sys.path.insert(0, os.path.abspath(".."))

from flask_image_search import __about__  # noqa: E402
from pallets_sphinx_themes import ProjectLink  # noqa: E402

# -- Project information -----------------------------------------------------

project = "Flask-Image-Search"
copyright = "2020 Hanan Fokkens"
author = "Hanan Fokkens"
release = __about__.__version__
version = __about__.__version__

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.viewcode",
    "sphinxcontrib.log_cabinet",
    "pallets_sphinx_themes",
]

autodoc_typehints = "description"
autosectionlabel_prefix_document = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "flask": ("https://flask.palletsprojects.com/", None),
    "sqlalchemy": ("https://docs.sqlalchemy.org/en/latest/", None),
    "flask_sqlalchemy": ("https://flask-sqlalchemy.palletsprojects.com/", None),
    "PIL": ("https://pillow.readthedocs.io/en/stable/", None),
}

source_suffix = ".rst"
master_doc = "index"
language = "en"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- HTML output -------------------------------------------------------------

html_theme = "flask"
html_context = {
    "project_links": [
        ProjectLink("PyPI Releases", "https://pypi.org/project/Flask-Image-Search/"),
        ProjectLink("Source Code", "https://github.com/hananf11/flask_image_search"),
        ProjectLink("Issue Tracker", "https://github.com/hananf11/flask_image_search/issues"),
    ]
}
html_sidebars = {
    "**": ["project.html", "localtoc.html", "relations.html", "searchbox.html"],
}
html_title = f"Flask-Image-Search Documentation ({version})"
html_show_sourcelink = False

# -- LaTeX / man / texinfo ---------------------------------------------------

latex_documents = [
    (master_doc, "flask_image_search.tex", html_title, author, "manual"),
]
man_pages = [(master_doc, "flask_image_search", html_title, [author], 1)]
texinfo_documents = [
    (master_doc, "flask_image_search", html_title, author,
     "flask_image_search", "Image search for Flask + Flask-SQLAlchemy.",
     "Miscellaneous"),
]
