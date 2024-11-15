import pandas as pd
from imdb import IMDb
import sys
import os

sys.path.append(os.path.abspath("src/utils"))
from helpers import clean_ids, get_box_office


dataset = pd.read_csv('Data/movies_ratings.csv')

#Quick checks
n_checks = 3
for i in range(n_checks):
    print(f"The initial box-office was {dataset.iloc[i]['Movie box office revenue']}, we find with Imdb library {get_box_office(dataset.iloc[i]['imdb_id'])}")

# Comment:
# We are indeed getting more box-office data, and recovering the same value in cases where we already had data ! Let's now apply the function on the whole dataset.

#Get the right imdb_id by removing 'tt' in front of IDs
dataset['imdb_id'] = dataset['imdb_id'].apply(lambda x : clean_ids(x, 'tt'))

#Create a dic telling if 'Movie box office revenue' is a NaN or not
isnan_dic = dataset['Movie box office revenue'].isna()

box_offices = dataset.apply(lambda x: get_box_office(x['imdb_id']) if isnan_dic[x.name] else x['Movie box office revenue'], axis=1) #Try to get the box-office from Imdb only if we don't already have it
dataset.insert(len(dataset.columns), 'IMDB Box-office', box_offices)

dataset.to_csv('Data/movies_synopsis_comp.csv', index=False)
