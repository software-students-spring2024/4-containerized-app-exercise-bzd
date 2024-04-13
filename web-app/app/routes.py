from app import app, mongo
from flask import render_template, request, jsonify

@app.route('/')
def index():
    data = mongo.db.collection.find()  # 假设您已经有一个名为'collection'的集合
    return render_template('index.html', data=data)
