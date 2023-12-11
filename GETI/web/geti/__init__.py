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

    from .views import view, func_view
    from . import filter
    app.register_blueprint(view.bp)
    app.register_blueprint(func_view.bp_func)
    
    app.jinja_env.filters['datetime'] = filter.format_datetime
    app.jinja_env.filters['calculate'] = filter.calculate_total_page_cnt
    app.jinja_env.globals.update(
        enumerate = enumerate
    )

    return app