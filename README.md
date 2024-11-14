README 
# Title to find
Smarter Movie Recommendations: Personalizing Choices with Socially Aware Systems

```bash
# clone project
git clone https://github.com/epfl-ada/ada-2024-project-padawans.git
cd <project repo>

# install requirements
pip install -r requirements.txt
```


# Abstract (150 words)
Movie recommendation algorithms typically rely on users' viewing history and preferences from similar users. However, these systems mostly focus on internal data within the streaming platform. What about external factors, like societal events? Such data can offer valuable insights into what people might want to watch depending on what the population as a whole is living. In this project, we'll use Latent Dirichlet Allocation (LDA) to uncover deeper movie themes beyond genres, identifying hidden patterns in movie synopses. By analyzing these themes we can use it to recommend similar movies to users in a classic way but also we can explore whether certain topics become more prominent during specific societal events. Ultimately, this approach could enable recommendations that reflect the current state of society.

# Research questions
+ What themes can be recovered using LDA?
+ Are those reflecting the genres? Or labels from the MPST dataset?
+ Have movies with similar themes been produced in similar countries or at similar times?
+ Does the box office income of movies about certain themes change depending on the period the movie has been produced?
+ Do we observe trends after historical events?
+ If there is a rise in a theme after an event how does the theme relate to the sentiment associated with the event?
+ How do ratings for the same movie theme vary by period?
+ What are the differences between the ratings of different movie themes?
+ How do user ratings and movie box office revenue correlate?
+ Is it feasible to group movies by similar theme to propose to the viewer a movie they will likely enjoy?


# Proposed additional datasets
MPST: Movie Plot Synopses with Tags: to retrieve movies' synopsis (https://www.kaggle.com/datasets/cryptexcode/mpst-movie-plot-synopses-with-tags?resource=download)

World Important Events - Ancient to Modern: https://www.kaggle.com/datasets/saketk511/world-important-events-ancient-to-modern

# Methods
## Latent Dirichlet Allocation (LDA)
Latent Dirichlet Allocation (LDA) is a probabilistic model that assumes each document contains a mixture of multiple hidden topics. Rather than focusing on individual words, LDA identifies themes or topics that underlie the words in a collection of documents. It filters out words that are highly repetitive across documents, as they don't contribute to distinguishing the themes. To assess the quality of the model, we can evaluate its coherence score.

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
   
# Timeline
# Organization within the team: A list of internal milestones up until project Milestone P3.

# Questions for TAs (optional): Add here any questions you have for us related to the proposed project.
