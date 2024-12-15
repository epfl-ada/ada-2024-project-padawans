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

# Networks
import networkx as nx
from networkx.algorithms import bipartite
import matplotlib.lines as mlines

# Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from scipy.stats import ttest_1samp
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

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
def draw_network_genre(movie_df, topic_dic, threshold):
    df_copy = movie_df.copy()
    genre_array = movie_df['Movie genres'].apply(ast.literal_eval)
    df_copy['Movie genres'] = genre_array
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
    random_proba_sum = 0
    genre_proba = 0
    for topic in topic_list:
        curr = df_copy[df_copy['Main Topic'] == topic]
        topic_occ[i] = len(curr)
        avr_genre_number = np.mean(curr['Movie genres'].apply(lambda x : len(x)))
        random_proba_sum += avr_genre_number/len(genre_list)
        for genre in genre_list:
            isin = curr['Movie genres'].apply(lambda x : genre in x)
            #print(len(genre_list))
            genre_mean = np.mean(isin)
            genre_proba += genre_mean
            if genre_mean > threshold:
                edges.append((topic, genre))
        i += 1
    B.add_edges_from(edges)
    projected = bipartite.weighted_projected_graph(B, topic_list, ratio=False)
    weights = nx.get_edge_attributes(projected,'weight')
    pos = nx.spring_layout(projected, seed=6)
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
    return random_proba_sum/(len(topic_list) * len(genre_list)), genre_proba/(len(topic_list) * len(genre_list))

#Draw network with tags
def draw_network_tags(movie_df, topic_dic, threshold):
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
    random_proba_sum = 0
    genre_proba = 0
    for topic in topic_list:
        curr = df_copy[df_copy['Main Topic'] == topic]
        topic_occ[i] = len(curr)
        avr_genre_number = np.mean(curr['tags'].apply(lambda x : len(x)))
        for genre in genre_list:
            isin = curr['tags'].apply(lambda x : genre in x)
            genre_mean = np.mean(isin)
            genre_proba += genre_mean
            random_proba_sum += avr_genre_number/len(genre_list)
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
    return random_proba_sum/(len(topic_list) * len(genre_list)), genre_proba/(len(topic_list) * len(genre_list))

# Train and evaluate a Linear Regression model 
def train_and_evaluate(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    return rmse, r2, y_pred

# Process the textual feautres using TF-IDF vectorizers and PCA for better capture of relationships    
def process_text_features_with_pca(data, textual_features, max_features=50, pca_variance=0.95, random_state=42):

    # Initialize TF-IDF vectorizers for textual features
    vectorizers = {feature: TfidfVectorizer(max_features=max_features) for feature in textual_features}

    # Apply TF-IDF transformation to each textual column
    tfidf_transformed = {
        feature: vectorizers[feature].fit_transform(
            data[feature].fillna('').astype(str)  # Ensure all entries are strings and NaNs are replaced
        )
        for feature in textual_features
    }

    # Combine all TF-IDF vectors into a single sparse matrix
    tfidf_combined = hstack(list(tfidf_transformed.values()))

    # Apply PCA to TF-IDF features
    pca = PCA(n_components=pca_variance, random_state=random_state)
    tfidf_reduced = pca.fit_transform(tfidf_combined.toarray())

    return vectorizers, tfidf_combined, pca, tfidf_reduced


# Allow to get the PCA top contributers 
def get_pca_top_contributions(pca, vectorizers, num_components_to_inspect=3, top_n=10):
 
    # Get PCA loadings (how much each original feature contributes to each PCA component)
    pca_loadings = pca.components_

    # Combine all TF-IDF feature names from the vectorizers
    tfidf_feature_names = []
    for key in vectorizers:
        tfidf_feature_names.extend(vectorizers[key].get_feature_names_out())

    # Analyze the top contributing features for the specified PCA components
    top_contributions = {}
    for i in range(num_components_to_inspect):
        component_loadings = pca_loadings[i]
        # Get the indices of the top contributing features for this component
        top_indices = component_loadings.argsort()[-top_n:][::-1]
        top_features = [(tfidf_feature_names[j], component_loadings[j]) for j in top_indices]
        top_contributions[f"PCA Component {i+1}"] = top_features

    return top_contributions

# Train many times a random model to compare with our current model
def test_significance(X_combined_reduced, y, rmse_comb, r2_comb, n_iterations=100):
# Store RMSE and R² for shuffled models
    rmse_shuffled_list = []
    r2_shuffled_list = []

    # Perform multiple shufflings and evaluations
    for _ in range(n_iterations):
        X_combined_shuffled = X_combined_reduced.copy()
        np.random.shuffle(X_combined_shuffled)
    
        X_train_shuff, X_test_shuff, y_train_shuff, y_test_shuff = train_test_split(
        X_combined_shuffled, y, test_size=0.2, random_state=42
        )
    
        rmse_shuff, r2_shuff, _ = train_and_evaluate(X_train_shuff, X_test_shuff, y_train_shuff, y_test_shuff)
        rmse_shuffled_list.append(rmse_shuff)
        r2_shuffled_list.append(r2_shuff)

    # Compute differences in performance metrics (Combined Model - Shuffled Models)
    rmse_diff = np.array(rmse_shuffled_list) - rmse_comb
    r2_diff = np.array(r2_shuffled_list) - r2_comb

    # Perform t-tests for RMSE and R² differences
    t_test_rmse = ttest_1samp(rmse_diff, 0)
    t_test_r2 = ttest_1samp(r2_diff, 0)

    # Summarize results
    shuffled_results_summary = pd.DataFrame({
        'Metric': ['RMSE', 'R²'],
        'Combined Model': [rmse_comb, r2_comb],
        'Random Model (Mean)': [np.mean(rmse_shuffled_list), np.mean(r2_shuffled_list)],
        'Mean Difference (Combined - Random)': [np.mean(rmse_diff), np.mean(r2_diff)],
        'p-value': [t_test_rmse.pvalue, t_test_r2.pvalue]
    })

    print("Shuffled Model Performance with Statistical Significance", shuffled_results_summary)
    return rmse_shuffled_list, r2_shuffled_list
    

