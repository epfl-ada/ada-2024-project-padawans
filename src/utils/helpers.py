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

# Plot ratings
import plotly.graph_objects as go

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
def generate_interactive_wordcloud(file, target1, target2, output_html="interactive_wordclouds.html", top_n=20):
    # Extract the top N genres by frequency
    genre_counts = file[target1].explode().value_counts()
    top_genres = genre_counts.head(top_n).index

    # Create a Plotly figure
    fig = go.Figure()

    # Generate a word cloud for each genre
    for genre in top_genres:
        genre_synopsis = file[file[target1].apply(lambda x: genre in x if isinstance(x, list) else False)][target2]
        synopsis_text = " ".join(genre_synopsis.dropna().astype(str))

        # Generate the word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(synopsis_text)
        wordcloud_image = wordcloud.to_array()

        # Add a trace for the word cloud (initially invisible)
        fig.add_trace(go.Image(z=wordcloud_image, visible=False))

    # Make the first word cloud visible by default
    fig.data[0].visible = True

    # Create dropdown buttons
    dropdown_buttons = [ dict(label=genre,method="update",args=[{"visible": [i == idx for i in range(len(top_genres))]},{"title": f"Word Cloud for Genre: {genre}"}])
        for idx, genre in enumerate(top_genres)]

    # Update layout with dropdown menu
    fig.update_layout(
        updatemenus=[dict(active=0,buttons=dropdown_buttons,direction="down",x=0.01,xanchor="left",y=0.99,yanchor="top",)],
        title="Word Cloud for Genres",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=10, r=10, t=30, b=10),
    )

    # Save as an interactive HTML file
    fig.write_html(output_html)
    
    # Show the plot
    fig.show()

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


# Plot the ratings distribution across top genres
def plot_ratings_dropdown(file, genre_column, rating_column, top_n=20, output_html="ratings_dropdown.html"):
    # Get the top N genres by frequency
    top_genres = file[genre_column].value_counts().head(top_n).index

    # Initialize the Plotly figure
    fig = go.Figure()

    # Add a histogram for each of the top genres (initially invisible)
    for genre in top_genres:
        filtered_data = file[file[genre_column] == genre]
        fig.add_trace(go.Histogram(
            x=filtered_data[rating_column],
            name=genre,
            nbinsx=50,
            visible=False
        ))

    # Make the first genre visible by default
    fig.data[0].visible = True

    # Create dropdown menu for selecting genres
    dropdown_buttons = [
        dict(
            label=genre,
            method="update",
            args=[{"visible": [i == idx for i in range(len(top_genres))]},
                  {"title": f"Ratings Distribution for Genre: {genre}"}])
        for idx, genre in enumerate(top_genres)
    ]

    # Update layout so that dropdown placed in the top-left corner
    fig.update_layout(
        updatemenus=[dict(active=0,buttons=dropdown_buttons,direction="down",x=0.01,xanchor="left",y=0.99,yanchor="top",)],
        title=f"Ratings Distribution Across Top {top_n} Genres",
        xaxis_title="IMDB Ratings",
        yaxis_title="Frequency"
    )

    # Save the plot as an HTML file
    fig.write_html(output_html)

    # Show the plot
    fig.show()


# Train and evaluate a Linear Regression model 
def train_and_evaluate(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    return rmse, r2, y_pred, model


# Give the most important features in each categories
def analyze_feature_importance(textual_features, tfidf_matrices, tfidf_vectorizers, model, rmse, r2, numerical_features=None, numerical=False):
    results = {}
    start_idx = 0

    # Analyze textual features
    for feature in textual_features:
        n_features = tfidf_matrices[feature].shape[1]
        coefficients = model.coef_[start_idx:start_idx + n_features]
        feature_names = tfidf_vectorizers[feature].get_feature_names_out()

        # Get the top 10 positive and negative features
        sorted_indices = np.argsort(coefficients)
        top_positive = [(feature_names[i], coefficients[i]) for i in sorted_indices[::-1][:10]]
        top_negative = [(feature_names[i], coefficients[i]) for i in sorted_indices[:10]]

        results[feature] = {
            "Top Positive Features": top_positive,
            "Top Negative Features": top_negative
        }
        start_idx += n_features

    # Analyze numerical features if enabled
    numerical_feature_importance = []
    if numerical and numerical_features is not None:
        numerical_coefficients = model.coef_[start_idx:]
        numerical_feature_importance = list(zip(numerical_features, numerical_coefficients))
        results["Numerical Features"] = numerical_feature_importance

    # Output results and evaluation metrics
    print(f"Mean Squared Error (MSE): {rmse}")
    print(f"R-squared (R²): {r2}")
    for feature, analysis in results.items():
        if feature != "Numerical Features":
            print(f"\nFeature: {feature}")
            print("Top Positive Features:")
            for term, coef in analysis["Top Positive Features"]:
                print(f"  {term}: {coef:.4f}")
            print("Top Negative Features:")
            for term, coef in analysis["Top Negative Features"]:
                print(f"  {term}: {coef:.4f}")
        else:
            print("\nNumerical Feature Importance:")
            for num_feature, coef in numerical_feature_importance:
                print(f"  {num_feature}: {coef:.4f}")

    return results


def test_significance(X_combined_reduced, y, rmse_comb, r2_comb, n_iterations=100):
# Store RMSE and R² for shuffled models
    rmse_shuffled_list = []
    r2_shuffled_list = []
    X_combined_reduced = X_combined_reduced.tocsr()

    # Perform multiple shufflings and evaluations
    for _ in range(n_iterations):
        indices = np.arange(X_combined_reduced.shape[0])  # Get row indices
        np.random.shuffle(indices)  # Shuffle the indices
        X_combined_shuffled = X_combined_reduced[indices]  # Reorder rows using the shuffled indices

    
        X_train_shuff, X_test_shuff, y_train_shuff, y_test_shuff = train_test_split(
        X_combined_shuffled, y, test_size=0.2, random_state=42
        )
    
        rmse_shuff, r2_shuff, _, _ = train_and_evaluate(X_train_shuff, X_test_shuff, y_train_shuff, y_test_shuff)
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

def draw_bipartite(movie_df, topic_dic, threshold, mode='genres'):
    """
    Draws a bipartite graph between topics and genres, only including genres with connections.

    Parameters:
        movie_df (DataFrame): DataFrame containing movie data, including 'Movie genres' and 'Main Topic'.
        topic_dic (dict): Dictionary of topics.
        threshold (float): Threshold to determine connections.

    Returns:
        tuple: Average random probability and genre probability.
    """
    # Step 1: Prepare data
    df_copy = movie_df.copy()

    if mode == "genres":
        genre_array = df_copy['Movie genres'].apply(ast.literal_eval)
        plot_title = "Filtered Bipartite Graph: Topics and Connected Genres"
    elif mode == "tags":
        df_copy['tags'] = df_copy['tags'].apply(lambda x: [item.strip() for item in x.split(',')])
        genre_array = df_copy['tags']
        plot_title = "Filtered Bipartite Graph: Topics and Connected Tags"
    else:
        raise ValueError("Mode must be either 'genres' or 'tags'")

    df_copy['Movie genres'] = genre_array
    all_genres = genre_array.explode().tolist()
    genre_list = list(set(all_genres))
    topic_list = list(topic_dic.values())

    # Step 2: Build the bipartite graph
    B = nx.Graph()
    B.add_nodes_from(topic_list, bipartite=0)  # Topics
    B.add_nodes_from(genre_list, bipartite=1)  # Genres

    edges = []
    topic_occ = np.zeros(len(topic_list))

    # Step 3: Add edges based on threshold
    for i, topic in enumerate(topic_list):
        curr = df_copy[df_copy['Main Topic'] == topic]
        topic_occ[i] = len(curr)
        for genre in genre_list:
            genre_mean = np.mean(curr['Movie genres'].apply(lambda x: genre in x))
            if genre_mean > threshold:
                edges.append((topic, genre))

    B.add_edges_from(edges)

    # Step 4: Filter nodes to keep only connected genres
    connected_genres = {v for u, v in B.edges() if u in topic_list}
    connected_nodes = topic_list + list(connected_genres)
    B_filtered = B.subgraph(connected_nodes)

    # Step 5: Layout and visualization
    pos = nx.bipartite_layout(B_filtered, topic_list)
    plt.figure(figsize=(14, 14))

    # Node sizes and colors
    node_sizes = [800 if n in topic_list else 300 for n in B_filtered.nodes()]
    node_colors = ['lightgreen' if n in topic_list else 'lightblue' for n in B_filtered.nodes()]

    # Draw graph
    nx.draw(
        B_filtered, pos,
        with_labels=True,
        node_color=node_colors,
        node_size=node_sizes,
        edge_color='gray',
        width=1.5,
        font_size=13
    )

    plt.title(plot_title)
    plt.show()
    

