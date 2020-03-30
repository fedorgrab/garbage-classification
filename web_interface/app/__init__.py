# fmt: off
from flask import Flask
from flask_bootstrap import Bootstrap
from web_interface.config import Config

app = Flask(__name__, static_url_path="/static/")
app.config.from_object(Config)
Bootstrap(app)

from web_interface.app import routes
# fmt: on
