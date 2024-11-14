README 
# Title to find
Watch out for unexpected Movie recommendations

```bash
# clone project
git clone https://github.com/epfl-ada/ada-2024-project-padawans.git
cd <project repo>

# install requirements
pip install -r pip_requirements.txt
```


# Abstract (150 words)
Movie recommendation algorithms often use users' viewing histories to suggest similar content through a method known as Content-Based Filtering. Similarity can be defined in many ways, such as shared directors or box-office earnings. This project focuses on using movie synopses to assess similarity, employing Latent Dirichlet Allocation (LDA) for unsupervised clustering. LDA is utilized to uncover hidden themes within movie synopses that extend beyond conventional genres, revealing deeper patterns in movie content. By analyzing these newly discovered themes, we aim to identify unique connections between films, enabling enhanced and diversified movie recommendations that aren't solely genre-dependent.


# Research questions
+ What themes can be recovered using LDA?
+ Were new themes/topics discovered? Are those reflecting the genres? Or labels from the MPST dataset?
+ Are themes reflecting countries or epoques/should the country be taken into account?
+ What are the differences between the ratings of different movie themes?
+ Should movies with higher ratings or box office earnings be given more importance when recommending similar films?

*Note: If the coherence score is low the rest of the analysis will remain valid using the label from the MPST dataset*

# Proposed additional datasets
MPST: Movie Plot Synopses with Tags: to retrieve movies' synopsis (https://www.kaggle.com/datasets/cryptexcode/mpst-movie-plot-synopses-with-tags?resource=download)

IMDb ratings: to get movie ratings (https://developer.imdb.com/non-commercial-datasets/)


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
To ensure the successful completion of the project within the remaining three weeks, the following timeline has been established:

### Week 1: Data Preparation and Model Implementation
+ Download and organize datasets: Ensure that MPST movie synopses, IMDb ratings, and optional historical event data are stored correctly.
+ Preprocess movie synopses: Clean and prepare the data by removing stop words, names, and any irrelevant content.
+ Implement the LDA model: Apply the model to the preprocessed data and experiment with different numbers of topics.
+ Optimize for coherence scores: Identify the most coherent topics and interpret them.
### Week 2: Analysis and Cross-Metadata Evaluation
+ Identify and analyze discovered themes: Determine if these themes align with or provide new insights beyond existing genres and tags.
+ Cross-analyze with metadata: Evaluate movie ratings and box office revenue to see if they have an impact on theme discovery.
+ Conduct temporal and regional analysis: Analyze trends over time and across different regions for the discovered themes.
### Week 3: Sentiment Analysis and Final Reporting
+ Perform sentiment analysis: Compare the sentiment of identified themes with the impact data from historical events (if applicable).
+ Summarize insights: Assess model quality and compile the main findings related to new themes and their implications.
+ Prepare the final report and presentation: Highlight the project’s outcomes, insights, and recommendations for enhancing movie recommendations.


# Organization within the team: A list of internal milestones up until project Milestone P3.

*Note: Portions of the text, were reformulated using ChatGPT to enhance clarity and readability.*
# Questions for TAs (optional): Add here any questions you have for us related to the proposed project.
