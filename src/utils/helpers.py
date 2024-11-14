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


def remove_named_entities(text):
        # Load the spaCy model to remove named entities
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        return ' '.join([token.text for token in doc if token.ent_type_ != "PERSON"])  # Exclude PERSON entities


def synopses_processing(sentences):
    # Process a list of sentences by removing named entities, filtering out stop words and converting words to lowercase

    stop_words = set(stopwords.words('english'))
    
    # Remove named entities
    cleaned_sentences= [remove_named_entities(doc) for doc in sentences]
    print("After removing names their are", len(cleaned_sentences), "sentences")
    
    # Remove stop words and lowercase word
    processed_sentences = [
        [word for word in word_tokenize(sentence.lower()) if word.isalnum() and word not in stop_words]
        for sentence in cleaned_sentences
    ]

    return processed_sentences

