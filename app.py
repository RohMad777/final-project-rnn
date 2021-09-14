import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create app
app = Flask(__name__)
model = pickle.load(open("model/model_knn_clf.pkl", "rb"))


@app.route("/")
def Home():
    return render_template("home.html")


@app.route("/predict", methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)

    output = {0: "Clients are not credible", 1: "Credible clients"}

    return render_template("prediction.html",
                           prediction_text="{}".format(output[prediction[0]]))


@app.route('/prediction')
def prediction():
    return render_template('prediction.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')


if __name__ == "__main__":
    app.run(debug=True)
