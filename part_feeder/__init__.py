from flask import Flask
from . import feeder
from .feeder import init_feeder
from .gallery import init_gallery


def create_app():
    # create and configure the app
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///persist/parts.db'

    from .db import set_db
    set_db(app)

    @app.route('/part-feeder/')
    def root():
        return app.send_static_file('index.html')

    from .db import get_db

    @app.route('/part-feeder/len/')
    def num_obs():
        _, Part = get_db()
        return {'len': Part.query.count()}
        # return {'len': len(feeder.displays)}

    with app.app_context():
        app = init_feeder(app)
        app = init_gallery(app)

    return app
