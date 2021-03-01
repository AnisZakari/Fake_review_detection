import pandas as pd
from joblib import dump, load
import re
from spacy.lang.fr.stop_words import STOP_WORDS
import fr_core_news_sm
nlp = fr_core_news_sm.load()
import tensorflow as tf
from keras.models import load_model
import numpy as np
import keras

@tf.function
def macro_soft_f1(y, y_hat):
    """Compute the macro soft F1-score as a cost (average 1 - soft-F1 across all labels).
    Use probability values instead of binary predictions.
    
    Args:
        y (int32 Tensor): targets array of shape (BATCH_SIZE, N_LABELS)
        y_hat (float32 Tensor): probability matrix from forward propagation of shape (BATCH_SIZE, N_LABELS)
        
    Returns:
        cost (scalar Tensor): value of the cost function for the batch
    """
    y = tf.cast(y, tf.float32)
    y_hat = tf.cast(y_hat, tf.float32)
    tp = tf.reduce_sum(y_hat * y, axis=0)
    fp = tf.reduce_sum(y_hat * (1 - y), axis=0)
    fn = tf.reduce_sum((1 - y_hat) * y, axis=0)
    soft_f1 = 2*tp / (2*tp + fn + fp + 1e-16)
    cost = 1 - soft_f1 # reduce 1 - soft-f1 in order to increase soft-f1
    macro_cost = tf.reduce_mean(cost) # average on all labels
    return macro_cost

keras.losses.custom_loss = macro_soft_f1


# Load Tools to preprocess and extract topics for function get_prev
#---------------------------------------#
lr = load('model_files/main_model.pkl')
preprocessor = load('model_files/preprocessor.pkl')
text_vectorizer = load('model_files/text_vectorizer.pkl')
topic_extractor = load('model_files/topic_extractor.pkl')
#---------------------------------------#


# Load Tools to preprocess and extract topics for function get_prev_full
#---------------------------------------#
model = tf.keras.models.load_model('model_files/tensorflow/model', custom_objects={'macro_soft_f1': macro_soft_f1})
preprocessor = load('model_files/tensorflow/preprocessor.pkl')
text_vectorizer = load('model_files/tensorflow/vectorizer.pkl')
topic_extractor = load('model_files/tensorflow/topic_extract.pkl')

superlatifs = ['bon', 'bonne', 'bons', 'bonnes', 'meilleur', 'meilleure', 'meilleures',
'mauvais', 'mauvaise', 'pire','petit', 'petite', 'moindre','plus', 'mieux', 'gros',
'impossible', 'totalement', 'loin', 'absence', 'très', 'tres', 'décu', 'decu',
'trop', 'jamais', 'toujours', 'aucun', 'déplorable', 'éviter', 'eviter', 'absolument',
'infect', 'infecte', 'fuir', 'fuire']
#---------------------------------------#

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
    meta.loc[:, 'len_review'] = pd.cut(meta.loc[:, 'len_review'], bins=[5.,  368., 4998.], labels = ['low', 'high'], include_lowest=True)
    
    #exclam
    meta.loc[:, 'exclam_count'] = pd.cut(meta.loc[:, 'exclam_count'], bins=[0.,   1,   3., 133.],labels=['low', 'high', 'very_high'], include_lowest=True)
    
    #uppercase
    meta.loc[:, 'upper_word_count'] = pd.cut(meta.loc[:, 'upper_word_count'], bins=[0.,   1,   2, 147.], labels = ['low', 'mid', 'high'], include_lowest=True)
    

    #We concatenate the topic and the metadata 
    conc = pd.concat([topic, meta], axis = 1)

    #preprocessing 
    data = preprocessor.transform(conc)
    prev = lr.predict(data)
    proba = lr.predict_proba(data)

    return prev[0], proba[0][0], proba[0][1],




preprocessor = load('model_files/tensorflow/preprocessor.pkl')
text_vectorizer = load('model_files/tensorflow/vectorizer.pkl')
topic_extractor = load('model_files/tensorflow/topic_extract.pkl')

superlatifs = ['bon', 'bonne', 'bons', 'bonnes', 'meilleur', 'meilleure', 'meilleures',
'mauvais', 'mauvaise', 'pire','petit', 'petite', 'moindre','plus', 'mieux', 'gros',
'impossible', 'totalement', 'loin', 'absence', 'très', 'tres', 'décu', 'decu',
'trop', 'jamais', 'toujours', 'aucun', 'déplorable', 'éviter', 'eviter', 'absolument',
'infect', 'infecte', 'fuir', 'fuire']

def get_prev_full(text, photos_for_review, rating, user_friends_count,
user_reviews_count, restaurant_average_rating, restaurant_reviews_count,
threshold = 0.5):

    #input data
    photos_for_review = photos_for_review
    rating = rating
    user_friends_count = user_friends_count
    user_reviews_count = user_reviews_count
    restaurant_average_rating = restaurant_average_rating
    restaurant_reviews_count = restaurant_reviews_count

    #Cleaning text
    review = text.replace('\n', ' ')
    review_clean = re.sub(r"[^A-zÀ-ÿ0-9' ]+", " ", review).lower()
    review_clean = re.sub(' +', ' ',review_clean).strip()
    lemmatized = " ".join([token.lemma_ for token in nlp(review_clean)])

    #Feature Engineering
    text_length = len(review)
    exclam_count = len(''.join(ch for ch in review if ch =='!'))
    upper_word_count = sum(map(str.isupper, review.split()))
    punctuation = len(''.join(ch for ch in review if ch ==',' or ch =='.' or ch ==';' or ch ==':' or ch =='?' or ch.isnumeric() ))
    sup_count = len([word for word in review_clean.split() if word in superlatifs])
    euros = len(''.join(ch for ch in review if ch =='€'))
    if euros >0:
        euros = 1

    negation = len([word for word in review_clean.split() if word in ['n', 'ne']])
    isnum = len(''.join(ch for ch in review_clean if ch.isnumeric()) )
    if isnum >0:
        isnum = 1



    #Vectorization and Topic Extraction
    vectrorized_item = text_vectorizer.transform([lemmatized])
    topic = topic_extractor.transform(vectrorized_item)

    meta = np.array([[
            photos_for_review,
            rating, 
            user_friends_count,
            user_reviews_count,
            restaurant_average_rating,
            restaurant_reviews_count,
            text_length,
            exclam_count, 
            upper_word_count,
            punctuation,
            sup_count,
            euros,
            negation,
            isnum 
                ]])        

    #preprocessing metadata
    metadata = preprocessor.transform(meta)
     

    #We concatenate the topic and the metadata 
    conc = np.concatenate([topic, metadata], axis = 1)

    #prediction
    proba_of_fake = model.predict(conc)
    if proba_of_fake > threshold:
        prediction = 1
    else:
        prediction = 0

    return prediction, proba_of_fake[0][0]