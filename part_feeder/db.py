from flask_sqlalchemy import SQLAlchemy

db: SQLAlchemy = None
Part = None


def set_db(app):
    global db, Part
    db = SQLAlchemy(app)

    class P(db.Model):
        id = db.Column(db.Integer, primary_key=True)
        points = db.Column(db.String, nullable=False)

        def __repr__(self):
            return self.points

    Part = P
    db.create_all()


def get_db():
    return db, Part
