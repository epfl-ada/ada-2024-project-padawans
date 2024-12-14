#Libraries 
import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt


# LDA
import spacy
# gensim is a popular library for topic modelling
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from imdb import IMDb
import re

# Wordcloud intereactive plor
from wordcloud import WordCloud
from ipywidgets import interact, widgets

#Networks
import networkx as nx
from networkx.algorithms import bipartite
import matplotlib.lines as mlines

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


# Function to generate word cloud for a specific genre
def generate_wordcloud(genre, file, target1, target2):
    genre_synopsis = file[file[target1].apply(lambda x: genre in x if isinstance(x, list) else False)][target2]
    synopsis_text = " ".join(genre_synopsis.dropna().astype(str))
    
    # Generate a word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(synopsis_text)
    
    # Plot the word cloud
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for {genre}', fontsize=16)
    plt.show()

#Draw network with movie genres
def draw_network_genre(movie_df, threshold):
    genre_array = movie_df['Movie genres'].apply(ast.literal_eval)
    # Get the list of genres
    all_genres = genre_array.explode().tolist()
    genre_list = list(set(all_genres))
    topic_list = topic_dic.values()

    B = nx.Graph()
    B.add_nodes_from(topic_list, bipartite=0)
    B.add_nodes_from(genre_list, bipartite=1)
    edges = []
    topic_occ = np.zeros(len(topic_list))

    #Iterate through all topics and then all genres
    #For a topic, if on average a genre is more present than the threshold, an edge is added between the topic and the genre
    i = 0
    for topic in topic_list:
        curr = movie_df[movie_df['Main Topic'] == topic]
        topic_occ[i] = len(curr)
        for genre in genre_list:
            isin = curr['Movie genres'].apply(lambda x : genre in x)
            genre_mean = np.mean(isin)
            if genre_mean > threshold:
                edges.append((topic, genre))
        i += 1
    B.add_edges_from(edges)
    projected = bipartite.weighted_projected_graph(B, topic_list, ratio=False)
    weights = nx.get_edge_attributes(projected,'weight')
    pos = nx.spring_layout(projected, seed=7)
    scaled_weights = [1.3 * weights[edge] for edge in projected.edges()]
    node_sizes = 10000 * topic_occ/topic_occ.sum()
    plt.figure(figsize=(12,12))
    nx.draw_networkx_nodes(
        projected, pos,
        node_color='lightblue', node_size=node_sizes,
        alpha=0.9 
    )
    i = 0
    for node, (x, y) in pos.items():
        label = str(node) 
        plt.text(
            x, y, label,
            weight='bold',
            fontsize=6*(1 + node_sizes[i]/5000), 
            horizontalalignment='center', verticalalignment='center'
        )
        i += 1
    nx.draw_networkx_edges(
        projected, 
        pos, 
        edge_color='grey', 
        width=scaled_weights,  # Set edges width proportional to the number of common genres
        alpha=0.55 
    )
    pourcent1 = 10000 * 0.01
    pourcent5 = 10000 * 0.05
    pourcent10 = 10000 * 0.1
    
    edge1 = 1
    edge5 = 5
    legend_elements = [
        plt.Line2D([0], [0], linestyle="none", marker="o", color='lightblue', markersize=2*np.sqrt(pourcent1/np.pi),
                   label="1%"),
        plt.Line2D([0], [0], linestyle="none", marker="o", color='lightblue', markersize=2*np.sqrt(pourcent5/np.pi),
                   label="5%"), 
        plt.Line2D([0], [0], linestyle="none", marker="o", color='lightblue', markersize=2*np.sqrt(pourcent10/np.pi),
                   label="10%"),         
        mlines.Line2D([], [], color='grey', linewidth=1.3 * edge1,
                      label="1"),
        mlines.Line2D([], [], color='grey', linewidth=1.3 * edge5,
                      label="5")
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=10, labelspacing=2.5, frameon=True)

#Draw network with tags
def draw_network_tags(movie_df, threshold):
    df_copy = movie_df.copy()
    df_copy['tags'] = movie_df['tags'].apply(lambda x : [item.strip() for item in x.split(',')])
    genre_array = df_copy['tags']
    # Get the list of genres
    all_genres = genre_array.explode().tolist()
    genre_list = list(set(all_genres))
    topic_list = topic_dic.values()

    B = nx.Graph()
    B.add_nodes_from(topic_list, bipartite=0)
    B.add_nodes_from(genre_list, bipartite=1)
    edges = []
    topic_occ = np.zeros(len(topic_list))

    #Iterate through all topics and then all genres
    #For a topic, if on average a genre is more present than the threshold, an edge is added between the topic and the genre
    i = 0
    for topic in topic_list:
        curr = df_copy[movie_df['Main Topic'] == topic]
        topic_occ[i] = len(curr)
        for genre in genre_list:
            isin = curr['tags'].apply(lambda x : genre in x)
            genre_mean = np.mean(isin)
            if genre_mean > threshold:
                edges.append((topic, genre))
        i += 1
    B.add_edges_from(edges)
    projected = bipartite.weighted_projected_graph(B, topic_list, ratio=False)
    weights = nx.get_edge_attributes(projected,'weight')
    pos = nx.spring_layout(projected, seed=7)
    scaled_weights = [1.3 * weights[edge] for edge in projected.edges()]
    node_sizes = 10000 * topic_occ/topic_occ.sum()
    plt.figure(figsize=(12,12))
    nx.draw_networkx_nodes(
        projected, pos,
        node_color='lightblue', node_size=node_sizes,
        alpha=0.9 
    )
    i = 0
    for node, (x, y) in pos.items():
        label = str(node) 
        plt.text(
            x, y, label,
            weight='bold',
            fontsize=6*(1 + node_sizes[i]/5000), 
            horizontalalignment='center', verticalalignment='center'
        )
        i += 1
    nx.draw_networkx_edges(
        projected, 
        pos, 
        edge_color='grey', 
        width=scaled_weights,  # Set edges width proportional to the number of common genres
        alpha=0.55 
    )
    pourcent1 = 10000 * 0.01
    pourcent5 = 10000 * 0.05
    pourcent10 = 10000 * 0.1
    
    edge1 = 1
    edge5 = 5
    legend_elements = [
        plt.Line2D([0], [0], linestyle="none", marker="o", color='lightblue', markersize=2*np.sqrt(pourcent1/np.pi),
                   label="1%"),
        plt.Line2D([0], [0], linestyle="none", marker="o", color='lightblue', markersize=2*np.sqrt(pourcent5/np.pi),
                   label="5%"), 
        plt.Line2D([0], [0], linestyle="none", marker="o", color='lightblue', markersize=2*np.sqrt(pourcent10/np.pi),
                   label="10%"),         
        mlines.Line2D([], [], color='grey', linewidth=1.3 * edge1,
                      label="1"),
        mlines.Line2D([], [], color='grey', linewidth=1.3 * edge5,
                      label="5")
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=10, labelspacing=2.5, frameon=True)



