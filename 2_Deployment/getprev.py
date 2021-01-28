import pandas as pd
from flask import Flask, url_for, request, jsonify, json
from joblib import dump, load
from werkzeug.exceptions import HTTPException
import re
import spacy
from spacy.lang.fr.stop_words import STOP_WORDS
import fr_core_news_sm
nlp = fr_core_news_sm.load()


# Load Tools to preprocess and extract topics
lr = load('model_files/main_model.pkl')
preprocessor = load('model_files/preprocessor.pkl')
text_vectorizer = load('model_files/text_vectorizer.pkl')
topic_extractor = load('model_files/topic_extractor.pkl')



def get_prev(text):
    #Cleaning
    review = text.replace('\n', ' ')
    #Feature Engineering
    length = len(review)
    exclam_count = len(''.join(ch for ch in review if ch =='!'))
    uppercase_word_count = sum(map(str.isupper, review.split()))

    meta = pd.DataFrame([[length, exclam_count, uppercase_word_count]], columns = ['len_review', 'exclam_count', 'upper_word_count'])
    #Cleaning, Vectorization and Topic Extraction

    review_clean = re.sub(r"[^A-zÀ-ÿ0-9' ]+", " ", review).lower()
    review_clean = re.sub(' +', ' ',review_clean).strip()
    lemmatized = " ".join([token.lemma_ for token in nlp(review_clean) if token.lemma_ not in STOP_WORDS])

    vectrorized_item = text_vectorizer.transform([lemmatized])
    topic = topic_extractor.transform(vectrorized_item)

    topic = pd.DataFrame(topic)

    #####DOUBLE CHECK BINS

    #Categorizing meta values, the bins are the quantiles used to cut the original dataset (see meta_data_analysis notebook)
    #length
    meta['len_review'] = pd.cut(meta['len_review'], bins=[   5.,  368., 4998.], labels = ['low', 'high'], include_lowest=True)
    #uppercase
    meta['upper_word_count'] = pd.cut(meta['upper_word_count'], bins=[  0.,   1,   2, 147.], labels = ['low', 'mid', 'high'],           include_lowest=True)
    #exclam
    meta['exclam_count'] = pd.cut(meta['exclam_count'], bins=[  0.,   1,   3., 133.],labels=['low', 'high', 'very_high'], include_lowest=True)

    #We concatenate the topic and the metadata 
    conc = pd.concat([topic, meta], axis = 1)

    #preprocessing 
    data = preprocessor.transform(conc)
    prev = lr.predict(data)
    proba = lr.predict_proba(data)

    return prev[0], proba[0][0], proba[0][1], conc[['len_review','exclam_count','upper_word_count']]
