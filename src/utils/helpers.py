#Libraries 
import numpy as np
import pandas as pd
import ast

# LDA
import spacy
# gensim is a popular library for topic modelling
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from imdb import IMDb
import re

#Functions
def replace_empty_with_nan(value):
    # Replace empty lists with NaN
    if isinstance(value, list) and len(value) == 0:
        return np.nan
    return value

def convert_to_list(df, column_name):
    #Convert to dictionnary 
    df[column_name] = df[column_name].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    #Convert to list
    df[column_name] = df[column_name].apply(lambda x: list(x.values()) if isinstance(x, dict) else None)
    return df[column_name]


def extract_year(value):
    # Check if the value is a year (4-digit number) and return it as is
    if isinstance(value, int) or (isinstance(value, str) and value.isdigit() and len(value) == 4):
        return int(value)
    # Try to convert complete date strings to datetime and extract the year
    try:
        return pd.to_datetime(value).year
    except (ValueError, TypeError):
        return np.nan  # Return NaN if conversion fails
    
    
def clean_ids(s, pattern=' '): #remove in string a pattern
    s = str(s).replace(pattern, '')
    return s
    

def get_box_office(imdb_ID): #return the box-office revenue for a specific imdb_id
    ia = IMDb()
    movie = ia.get_movie(imdb_ID)
    box_office = movie.get('box office')
    if box_office and 'Cumulative Worldwide Gross' in box_office:
        box_office_num = float(re.sub(r'[^\d.]', '', box_office['Cumulative Worldwide Gross']))
        #box_office_num = float(box_office['Cumulative Worldwide Gross'].replace('$', '').replace(',', ''))
        return box_office_num
    return None  # Return None if no box office data is found

def get_intra_similarity(word_list, model_w2v):
    length = len(word_list)
    intra_sim = 0
    for i in range(length):
        for j in range(length):
            if i != j:
                if word_list[i] in model_w2v and word_list[j] in model_w2v:
                    intra_sim += model_w2v.similarity(word_list[i], word_list[j])/(length * (length-1))
    return intra_sim

def get_inter_similarity(word_list1, word_list2, model_w2v):
    length1 = len(word_list1)
    length2 = len(word_list2)
    inter_sim = 0
    for i in range(length1):
        for j in range(length2):
            if word_list1[i] in model_w2v and word_list2[j] in model_w2v:
                inter_sim += model_w2v.similarity(word_list1[i], word_list2[j])/(length1 * length2)
    return inter_sim

def get_similarity(mod, model_w2v):
    word_by_topics = []
    for _, topic_words in mod.show_topics(num_topics=-1, num_words=10, formatted=False):
        words = [word for word, _ in topic_words]  # Extract just the words
        word_by_topics.append(words)
    nbr_topics = len(word_by_topics)
    coherence_score = []
    for i in range(nbr_topics):
        for j in range(nbr_topics):
            if i != j:
                intra_similarity_i = get_intra_similarity(word_by_topics[i], model_w2v)
                intra_similarity_j = get_intra_similarity(word_by_topics[j], model_w2v)
                inter_similarity = get_inter_similarity(word_by_topics[i], word_by_topics[j], model_w2v)
                coherence_score.append((intra_similarity_i + intra_similarity_j) / (2 * inter_similarity))
    return np.mean(coherence_score)



