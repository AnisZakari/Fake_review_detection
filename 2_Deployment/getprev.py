import pandas as pd
from joblib import dump, load
import re
import tensorflow as tf
#from keras.models import load_model
import numpy as np
#import keras

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

@tf.function
def f1_score(y_true, y_pred):
    return 1



tf.keras.losses.custom_loss = macro_soft_f1

# Load Tools to preprocess and extract topics for function get_prev_full
#---------------------------------------#
model = tf.keras.models.load_model('model_files/tensorflow/model', custom_objects={'macro_soft_f1': macro_soft_f1, 'f1_score': f1_score })
preprocessor = load('model_files/tensorflow/data_scaler.pkl')
text_vectorizer = load('model_files/tensorflow/text_vectorizer.pkl')
topic_extractor = load('model_files/tensorflow/topic_extractor.pkl')

cleaner = load('model_files/tensorflow/cleaner.pkl')
expand_text = load('model_files/tensorflow/expand_text.pkl')
fix_fancy_words = load('model_files/tensorflow/fix_fancy_words.pkl')

superlatifs = ['bon', 'bonne', 'bons', 'bonnes', 'meilleur', 'meilleure', 'meilleures',
'mauvais', 'mauvaise', 'pire','petit', 'petite', 'moindre','plus', 'mieux', 'gros',
'impossible', 'totalement', 'loin', 'absence', 'très', 'tres', 'décu', 'decu',
'trop', 'jamais', 'toujours', 'aucun', 'déplorable', 'éviter', 'eviter', 'absolument',
'infect', 'infecte', 'fuir', 'fuire']

def get_prev_full(text, photos_for_review, rating, user_friends_count,
user_reviews_count, restaurant_average_rating, restaurant_reviews_count, restaurant_expensiveness,
threshold = 0.5):

    #input data
    photos_for_review = photos_for_review
    rating = rating
    user_friends_count = user_friends_count
    user_reviews_count = user_reviews_count
    restaurant_average_rating = restaurant_average_rating
    restaurant_reviews_count = restaurant_reviews_count
    restaurant_expensiveness = restaurant_expensiveness

    #Feature Engineering
    text_length = len(review)
    punctuation_count = len(''.join(ch for ch in review if ch =='!' or ch == '?' or ch == '.' or ch =='%'))
    upper_word_count = sum(map(str.isupper, review.split() )) 
    average_word_length = np.mean([len(word) for word in review.split()])
    digits = len(''.join(ch for ch in review if ch.isnumeric())) 
    negation= len([word for word in review.split() if word in ['n', 'ne']])
    sup_count = len([word for word in review.split() if word in superlatifs])

    #Cleaning the text
    review = re.sub(r"[\n]", ' ', review)
    review = review.lower()
    review = expand_text(review)
    review = cleaner(review)
    review = re.sub(r" +"," ", review)
    review = "".join(ch for ch in review if ch.isalnum() or ch == '!' or ch =='?' or ch ==' ' or ch == '-' or ch =='%')

    #Vectorization and Topic Extraction
    vect_review = vectorizer.transform([review])
    topics = svd.transform(vect_review)

    #creating metadata vector
    meta_vector = np.array([[photos_for_review, rating, user_friends_count, user_reviews_count, restaurant_average_rating, restaurant_reviews_count, restaurant_expensiveness,
                text_length, punctuation_count, upper_word_count, average_word_length, digits, negation, sup_count]])
    #normalize data
    meta_vector = scaler.transform(meta_vector)
        
    #concatenate both to create input vector
    input_vector = np.concatenate([topics, meta_vector], axis = 1)

    #prediction
    proba_fake = model.predict(input_vector)[0][0]
    if proba_fake > threshold:
        prediction = 1
    else:
        prediction = 0

    return prediction, proba_fake