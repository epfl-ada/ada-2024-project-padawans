# Watch out for unexpected movie recommendations !

```bash
# clone project
git clone https://github.com/epfl-ada/ada-2024-project-padawans.git
cd <project repo>

# install requirements
pip install -r pip_requirements.txt
```


# Abstract (150 words)
Movie recommendation algorithms often use users' viewing histories to suggest similar content through a method known as Content-Based Filtering. Similarity can be defined in many ways, such as genres, shared directors or box-office earnings. This project focuses on using movie synopses to assess similarity, employing Latent Dirichlet Allocation (LDA) for unsupervised clustering. LDA is utilized to uncover hidden themes within movie synopses that extend beyond conventional genres, revealing deeper patterns in movie content. By analyzing these newly discovered themes, we aim to identify unique connections between films, enabling enhanced and diversified movie recommendations.


# Research questions
+ What topics can be recovered using LDA on the movie synopsis?
+ Do these topics reflect the genres? Or labels from the MPST dataset?
+ Are topics enough for movie recommendations?
+ What additional features should we consider?

*Note: If the coherence score is low the rest of the analysis will remain valid using the label from the MPST dataset*

# Proposed additional datasets
MPST: Movie Plot Synopses with Tags: to retrieve movies' synopsis (https://www.kaggle.com/datasets/cryptexcode/mpst-movie-plot-synopses-with-tags?resource=download)

IMDb ratings: to get movie ratings (https://developer.imdb.com/non-commercial-datasets/)


# Methods
## Latent Dirichlet Allocation (LDA)
Latent Dirichlet Allocation (LDA) is a probabilistic model that assumes each document contains a mixture of multiple hidden topics. Initially, LDA randomly assigns words to topics, kicking off a learning procedure. It then traverses through each word in each document, applying the complex formula presented in the source. Through numerous iterations of this procedure, it eventually yields a set of topics. Rather than focusing on individual words, LDA identifies themes or topics that underlie the words in a collection of documents. 
It filters out words that are highly repetitive across documents, as they don't contribute to distinguishing the themes. 

<img src="https://cdn.botpenguin.com/assets/website/Topic_Modeling_35bd15572c.webp" width="400" height="400">

+ Each topic is a distribution over words
+ Each document is a mixture of corpus wide topics
+ Each word is drawn from one of those topics


The users must decide on the amount of topics present in the document as well as interpret what the topics are.

To assess the quality of the model, we will evaluate its coherence score.

### Coherence score
Coherence measures how well the words grouped in a given topic are related in meaning and whether they frequently co-occur within the same document. A higher coherence score indicates that the words in a topic are semantically related and distinct from other topics, suggesting that the topic grouping is meaningful and accurate.

sources: 
https://medium.com/@pinakdatta/understanding-lda-unveiling-hidden-topics-in-text-data-9bbbd25ae162
https://medium.com/analytics-vidhya/latent-dirichelt-allocation-1ec8729589d4#:~:text=Latent%20Dirichlet%20Allocation%20(LDA)%20is%20a%20method%20for%20associating%20sentences,facts%20before%20applying%20these%20processes.

## Required External Files

Some files required for this project are too large to store in the repository (>25MB). Thus, once you have cloned the repository locally, please download the required files from the following sources:

1. **Movie Summaries**: [link to file](https://www.cs.cmu.edu/~ark/personas/data/MovieSummaries.tar.gz)  
   - Description: Original given movie dataset.
   - Location: Unzip the file and only keep `movie.metadata.tsv` and `plot_summaries` in the `Data/` folder.

2. **MPST: Movie Plot Synopses with Tags**: [link to file](https://www.kaggle.com/datasets/cryptexcode/mpst-movie-plot-synopses-with-tags?resource=download)  
   - Description: Contains movies' synopsis (and tags).
   - Steps to follow: Click on the Download button on the top right part of the screen to download the file. It will show you a sub-menu where you can click on 'Download dataset as zip' to download the file.
   - Location: Place this file in the `Data/` folder.

3. **Movie ratings dataset**: [link to file](https://datasets.imdbws.com/title.ratings.tsv.gz)  
   - Description: Contains movie ratings (mean) and the number of ratings received.
   - Location: Place this file in the `Data/` folder.

4. **Movies with IMDB Box Income** (movies_synopsis_comp.csv)
   - This file can be found in the `Data/` folder.
   - Description: Contains movie box incomes extracted with IMdB library

5. **Processed sentences for LDA**: [link to file](https://drive.google.com/file/d/1K_l2LZGIvGgbZ3Q-u0v3W-_dRlWlIpOM/view?usp=sharing)
   - Description: Contains sentences without stop words and names.
   - Location: Place this file in the `Data/` folder.
   
Make sure to follow the directory structure to ensure the project runs correctly.
   
# Timeline
1. Present different themes obtained with LDA, depending on the number of topics
2. Compare the themes obtained with the labels and the genres
3. Look at the characteristics (country, ratings, box-office) of the themes
4. Look at the temporal evolution of the themes

# Contribution of members
+ Chloé : Feature analysis
+ Clara : Creating the website
+ Esteban : Similarity Analysis
+ Camille : Recommendation Analysis, Graph Analysis
+ Thomas : Preprocessing data, Topics discovery, Graph Analysis

