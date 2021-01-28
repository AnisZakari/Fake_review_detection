from flask import Flask, render_template
import pandas as pd
import numpy as np
from flask import Flask, url_for, request, jsonify, json
from joblib import dump, load
from werkzeug.exceptions import HTTPException
from flask import Flask, url_for, request, jsonify, json
from getprev import get_prev

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/more_love")
def more_love():
    things_i_love = [
        "Star Wars",
        "Coffee",
        "Cookies",
    ]
    return render_template("more_love.html", things_i_love=things_i_love)



@app.route("/predict", methods=["POST"])
def predict():
    # Check parameters
    if request.json:
        # Get JSON as dictionnary
        json_input = request.get_json()
        value = [[json_input['review']]]
        label, proba_real, proba_fake, text_infos = get_prev(str(value))
        if label == 0:
            r = "Cette review à l'air légitime, avec une probabilité de " + str(round(proba_real,3))
        if label == 1:
            r = "Cette review est probablement fausse avec une probabilité de " + str(round(proba_fake,3))

        # Return prediction
        #response = {
            # Since prediction is a float and jsonify function can't handle
            # floats we need to convert it to string
        #    "prediction": str(r),
        #}
        
        return r


if __name__ == "__main__":
    app.run(debug=True)
