from flask import render_template
from app import app, db

# Route for the main index page
@app.route('/')
def index():
    # Fetch data from your MongoDB's collection
    results = db['precision'].find()  # Replace with your actual collection name
    return render_template('index.html', results=list(results))
