import numpy as np
from flask import Flask, request, render_template
from numpy.core.fromnumeric import argmax
import tensorflow as tf
import math

# Create app
app = Flask(__name__)
model = tf.keras.models.load_model('./save_model')


@app.route("/")
def Home():
    return render_template("home.html")


@app.route("/predict", methods=["POST"])
def predict():
    float_features = request.form['sentence']
    features = [float_features]
    prediction = model.predict(features)

    output = {0: "Negatif", 1: "Netral", 2: "Positif"}

    return render_template("predict.html",
                           prediction_text="{}".format(output[(int(
                               math.floor(prediction[0][1])))]))


@app.route('/data-set')
def dataSet():
    return render_template('tokopediadata.html')


@app.route('/predict')
def predicts():
    return render_template('predict.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')


if __name__ == "__main__":
    app.run(debug=True)
