from flask import Flask, render_template, url_for, request, jsonify, json
import pandas as pd
import numpy as np
from joblib import dump, load
from werkzeug.exceptions import HTTPException
from getprev import get_prev

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/dashboard")
def dash():
    return render_template("dashboard.html")

@app.route("/", methods=['POST'])
def my_form_post():
    value = request.form['text']
    len_value = len(''.join(ch for ch in str(value) if ch.isalnum()))

    if len_value < 10:

        r = "Veuillez écrire une review plus longue."
    
    else:
        label, proba_real, proba_fake = get_prev(str(value))
        if label == 0:
            r = "Cette review à l'air légitime, avec une probabilité de " + str(round(proba_real,3))
        if label == 1:
            r = "Cette review est probablement fausse avec une probabilité de " + str(round(proba_fake,3))

    return r





'''
@app.route('/backup', methods=['POST'])
def my_form_post():
    value = request.form['text']
    len_value = len(''.join(ch for ch in str(value) if ch.isalnum()))

    if len_value < 10:

        r = "Veuillez écrire une review plus longue."
    
    else:
        label, proba_real, proba_fake, text_infos = get_prev(str(value))
        if label == 0:
            r = "Cette review à l'air légitime, avec une probabilité de " + str(round(proba_real,3))
        if label == 1:
            r = "Cette review est probablement fausse avec une probabilité de " + str(round(proba_fake,3))

    return r
'''

if __name__ == "__main__":
    app.run(debug=True)
