import os

from flask import Flask
from . import feeder
from .feeder import init_feeder


def create_app():
    # create and configure the app
    app = Flask(__name__)

    @app.route('/part-feeder/')
    def root():
        return app.send_static_file('index.html')

    @app.route('/part-feeder/len/')
    def num_obs():
        return {'len': len(feeder.displays)}

    with app.app_context():
        app = init_feeder(app)

    return app