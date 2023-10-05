from flask import Flask, jsonify
import pickle
import os

app = Flask(__name__)

@app.route("/")
def hello():
    return("Hello from FLASK!")

@app.route('/predict/<int:Feature1>/<int:Feature2>')
def predict(Feature1, Feature2):
    with open('../../models/model.pkl', 'rb') as fd:
        clf = pickle.load(fd)
    prediction = int(clf.predict([[Feature1, Feature2]])[0])
    return(jsonify({'Target':prediction}))