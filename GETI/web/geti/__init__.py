from flask import Flask
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy

import os
import config

db = SQLAlchemy()
migrate = Migrate()

def create_app():
    app = Flask(__name__)
    app.config.from_object(config)
    app.secret_key = os.urandom(24)

    # ORM
    db.init_app(app)
    migrate.init_app(app, db)
    from . import models

    from .views import view
    app.register_blueprint(view.bp)

    return app