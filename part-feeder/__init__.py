import os

from flask import Flask
from .feeder import init_feeder


def create_app():
    # create and configure the app
    app = Flask(__name__)

    @app.route('/')
    def root():
        return app.send_static_file('index.html')

    with app.app_context():
        app = init_feeder(app)

    return app