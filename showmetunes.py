#!/Users/kelly/anaconda3/envs/dsi/bin/python

import sys
import os
import csv

import numpy as np
import pandas as pd
import spacy
import en_core_web_lg
from textblob import TextBlob

# Take in a string argument in ""
user_input = sys.argv

# Read in musical data and spaCy vectorization model for processing similarity
df = pd.read_csv('./data/musical_for_app.csv')
nlp = en_core_web_lg.load()

# Define a function to take in user input and output a spaCy doc
def convert_to_doc(user_input):
    doc = nlp(user_input)
    return doc

# Define a function to take in user input and output its TextBlob sentiment
def get_sentiment(user_input):
    return TextBlob(user_input).sentiment.polarity

# Define a function to find each musical summary's sentiment and respective cosine similarity with the user input
def similarity_and_sentiments(user_input): #, df=df, summ_col='docs', sentiment_col='sentiment', name_col='name'):

    # Convert user input to a spaCy doc
    user_input_doc = convert_to_doc(user_input)

    # Calculate similarity with each musical and create dicitonary of similarities and sentiments
    similarity_dict = {}
    for i, summ in enumerate(df['docs']):
        summ = nlp(summ)
        sim = user_input_doc.similarity(summ)
        similarity_dict[sim] = [df['sentiment'][i], df['name'][i]]

    return similarity_dict

# Define a function to sort the list musical similarities and pull out top ten
def top_ten(similarity_dict):

    # Sort the musicals by similarity
    in_order = sorted(similarity_dict.items())

    # Consider only top 10 most similar musicals before including sentiment
    num_to_consider = 10

    # Pull out top 10 most similar musicals
    sentiment_list = []
    for i in range(num_to_consider):
        sentiment_list.append(in_order[-num_to_consider:][i][1])

    return sentiment_list

# Define a function to
def get_recommendations(sentiment_list, user_input):

    # Calculate sentiment of user input
    user_sentiment = get_sentiment(user_input)

    # Pull out top 3 musicals with closest sentiment rating in either direction
    sentiment_differences = []
    for sentiment, musical in sentiment_list:
        diff = np.abs(user_sentiment - sentiment)
        sentiment_differences.append([diff, musical])

    # Extract musical names, in order
    top_three = sorted(sentiment_differences[:3])
    final_recommendations = []
    for sentiment, musical in top_three:
        final_recommendations.append(musical)

    return final_recommendations

# Define a function to take in user input and output 3 musical recommendations
def recommend(user_input):
    similarity_dict = similarity_and_sentiments(user_input)
    sentiment_list = top_ten(similarity_dict)
    final_recommendations = get_recommendations(sentiment_list, user_input)
    return final_recommendations

if __name__ == '__main__':
    user_input = sys.argv
    df = pd.read_csv('./data/musical_for_app.csv')
    nlp = en_core_web_lg.load()
    print(recommend(sys.argv))
