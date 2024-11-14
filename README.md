README 
# Title to find
Watch out for unexpected Movie recommendations

```bash
# clone project
git clone https://github.com/epfl-ada/ada-2024-project-padawans.git
cd <project repo>

# install requirements
pip install -r requirements.txt
```


# Abstract (150 words)
Movie recommendation algorithms typically rely on what users have previously watched to recommend similar content. This method is called Content-Based Filtering. The notion of similarity can vary from movie director to box-office revenue and so on. In this project, movies's synopses are used to assess similarities. We use Unsupervised clustering Latent Dirichlet Allocation (LDA) to uncover movie themes beyond genres, identifying patterns in movie synopses. By analyzing these new themes we can discover new similarities between movies and recommend them based on that. 



# Research questions
+ What themes can be recovered using LDA?
+ How many ways are there to efficiently cluster movie synopses? (Topics = 1 or 56?!)
+ Were new themes/topics discovered? Are those reflecting the genres? Or labels from the MPST dataset?
+ Are themes reflecting countries or epoques/should the country be taken into account?
+ What to do if 2 movies show similar levels of similarities?
+ Should movies with higher ratings or box office earnings be given more importance when recommending similar films?
+ Does the box office revenue of certain themes vary based on the period in which the movie was made?
+ How do theme ratings vary by period?
+ What are the differences between the ratings of different movie themes?
+ Should movies with higher ratings or box office earnings be given more importance when recommending similar films?


# Proposed additional datasets
MPST: Movie Plot Synopses with Tags: to retrieve movies' synopsis (https://www.kaggle.com/datasets/cryptexcode/mpst-movie-plot-synopses-with-tags?resource=download)

IMDb ratings: to get movie ratings (https://developer.imdb.com/non-commercial-datasets/)

(KEEP?) World Important Events - Ancient to Modern: https://www.kaggle.com/datasets/saketk511/world-important-events-ancient-to-modern



# Methods
## Latent Dirichlet Allocation (LDA)
Latent Dirichlet Allocation (LDA) is a probabilistic model that assumes each document contains a mixture of multiple hidden topics. Initially, LDA randomly assigns words to topics, kicking off a learning procedure. It then traverses through each word in each document, applying the formula discussed earlier. Through numerous iterations of this procedure, it eventually yields a set of topics. Rather than focusing on individual words, LDA identifies themes or topics that underlie the words in a collection of documents. 
It filters out words that are highly repetitive across documents, as they don't contribute to distinguishing the themes. 

The users must decide on the amount of topics present in the document as well as interpret what the topics are.

To assess the quality of the model, we will evaluate its coherence score.

![Alt text](https://miro.medium.com/v2/resize:fit:1178/format:webp/0*J1oMupf58psVRVCH.png)

sources: 
https://medium.com/@pinakdatta/understanding-lda-unveiling-hidden-topics-in-text-data-9bbbd25ae162
https://medium.com/analytics-vidhya/latent-dirichelt-allocation-1ec8729589d4#:~:text=Latent%20Dirichlet%20Allocation%20(LDA)%20is%20a%20method%20for%20associating%20sentences,facts%20before%20applying%20these%20processes.

### Coherence score
Coherence measures how well the words grouped in a given topic are related in meaning and whether they frequently co-occur within the same document. A higher coherence score indicates that the words in a topic are semantically related and distinct from other topics, suggesting that the topic grouping is meaningful and accurate.


## Worflow
1. Exploring the potential of LDA to uncover themes from movie synopses
   + Synopses preprocessing (Removing stop words, names, ...)
   + Test different number of topics
   + Analyse their coherence score
   + Analyse the most coherent topics
2. Analyzing the identified themes to determine if they provide new insights
   + Identify which movies are associated with each theme
   + For all movies with the same theme check the timeline and country of origin
3. What are the trend of each theme? (weight by box-office if possible)
   + Weight movies by their impacts (movies that have been watched more should be weighed more in the analysis)
   + Check for aggregate
   + After chosen events (see P2 analysis) check for rise in a genre/theme
4. Sentiment analysis using the impact column of the historical dataset
   + Retrieve sentiment associated with the event from impact column
   + Compare to theme sentiment
  
## Required External Files

Some files required for this project are too large to store in the repository (>25MB). Thus, once you have cloned the repository locally, please download the required files from the following sources:

1. **Movie Summaries**: [link to file](https://www.cs.cmu.edu/~ark/personas/data/MovieSummaries.tar.gz)  
   - Description: Original given movie dataset.
   - Location: Place this file and its contents in the `DATA/Raw/` folder and unzip its contents inside `DATA/Raw/MovieSummaries`.

2. **MPST: Movie Plot Synopses with Tags**: [link to file](https://www.kaggle.com/datasets/cryptexcode/mpst-movie-plot-synopses-with-tags?resource=download)  
   - Description: Contains movies' synopsis (and tags).
   - Steps to follow: Click on the Download button on the top right part of the screen to download the file. It will show you a sub-menu where you can click on 'Download dataset as zip (30 MB)' to download the file.
   - Location: Place this file in the `DATA/Raw/` folder and unzip its contents inside `DATA/Raw/archive`.

3. **Movie ratings dataset**: [link to file](https://datasets.imdbws.com/title.ratings.tsv.gz)  
   - Description: Contains movie ratings (mean) and the number of ratings received.
   - Location: Place this file in the `DATA/Raw/` folder and unzip its content inside.

Make sure to follow the directory structure to ensure the project runs correctly.

Note: All processed files will be stored in the `DATA/Processed/` directory.
   
# Timeline
# Organization within the team: A list of internal milestones up until project Milestone P3.

# Questions for TAs (optional): Add here any questions you have for us related to the proposed project.
