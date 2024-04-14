"""Define routes for the Flask application."""
from flask import render_template

def init_app(app, db):
    """pass app and db into routes."""
    @app.route('/')
    def index():
        """Render index page with results from the database."""
        results = db['precision'].find()
        return render_template('index.html', results=list(results))
 