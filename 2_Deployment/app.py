from flask import Flask, render_template, url_for, request, jsonify, json
import pandas as pd
import numpy as np
from joblib import dump, load
from werkzeug.exceptions import HTTPException
from getprev import get_prev_full
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
from wtforms.validators import DataRequired
from flask_bootstrap import Bootstrap

app = Flask(__name__)
Bootstrap(app)


@app.route("/")
def index():
    return render_template("index.html")
    
@app.route("/", methods=['POST'])
def get_prediction():

    review = request.form['review']
    len_value = len("".join(ch for ch in review))
    photos_for_review = request.form['photos_for_review']
    rating = request.form['rating']
    user_friends_count = request.form['user_friends_count']
    user_reviews_count = request.form['user_reviews_count']
    restaurant_average_rating = request.form['restaurant_average_rating']
    restaurant_reviews_count = request.form['restaurant_reviews_count']

    if photos_for_review == 'yes':
        photos_for_review = 2
    if photos_for_review == 'no':
        photos_for_review = 0

    #test = str(photos_for_review) + ' ' + str(rating) + ' ' + str(user_friends_count) + ' ' + str(user_reviews_count) + ' ' + str(restaurant_average_rating) + ' ' + str(restaurant_reviews_count)

    all_values_ok = (photos_for_review != '') & (rating != '' ) & (rating != '' ) & (user_friends_count != '' ) & (user_reviews_count != '' ) & (restaurant_average_rating != '' ) & (restaurant_reviews_count != '' )

    len_review = len(''.join(ch for ch in str(review) if ch.isalnum()))

    if len_value <= 10:

        r = "Veuillez écrire une review plus longue."
        label ='empty'

    
    if (all_values_ok == False) :
        r = "Veuillez renseigner toutes les informations."
        label ='empty'
    


    if (all_values_ok) & (len_value >10):
        label, proba_fake = get_prev_full(review, float(photos_for_review), float(rating), float(user_friends_count), float(user_reviews_count), float(restaurant_average_rating), float(restaurant_reviews_count), float(restaurant_expensiveness), threshold = 0.5)
        if label == 1:
            r = "Cette review est probablement fausse " # + "avec une probabilité de " + str(round(proba_fake,3))
        if label == 0:          
            r = "Cette review à l'air légitime "  #+ "avec une probabilité de " + str(round(1-proba_fake,3))
    return render_template("send_prediction.html", label = label, review = review, response = r)

@app.route("/dashboard")
def dash():
    return render_template("dashboard.html")

if __name__ == "__main__":
    app.run(debug=True)
