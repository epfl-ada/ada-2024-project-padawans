import pandas as pd
import numpy as np

# LDA
# gensim is a popular library for topic modelling
# pip install gensim
# pip install nltk
# pip install kagglehub
# pip install spacy
# pip install IMDbPY
from imdb import IMDb
import re

movies_synopsis = pd.read_csv('test_thomas.csv')

#Create a dic telling if 'Movie box office revenue' is a NaN or not
isnan_dic = movies_synopsis['Movie box office revenue'].isna()

def get_box_office(imdb_ID): #return the box-office revenue for a specific imdb_id
    ia = IMDb()
    movie = ia.get_movie(imdb_ID)
    box_office = movie.get('box office')
    if box_office and 'Cumulative Worldwide Gross' in box_office:
        box_office_num = float(re.sub(r'[^\d.]', '', box_office['Cumulative Worldwide Gross']))
        return box_office_num
    return None  # Return None if no box office data is found
print('It started')
new_box = movies_synopsis.apply(lambda x: get_box_office(x['imdb_id']) if isnan_dic[x.name] else x['Movie box office revenue'], axis=1)
movies_synopsis.insert(len(movies_synopsis.columns), 'IMDB Box-office', new_box)
movies_synopsis.to_csv('with_new_box_office.csv', index=False)
print('It finished')