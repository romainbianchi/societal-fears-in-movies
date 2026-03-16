# When Society's Greatest Fears hit the Big Screen

## Datastory
Discover how societal fears are depicted in the datastory through our website: [Datastory](https://octavioprofeta.github.io/ "datastory") 

## Abstract 
In this project, we will analyze the chronological evolution of movies’ themes in relation to society’s fears. Fears have evolved significantly over the decades, shifting from the apprehension of war in the mid-20th century to concerns about emerging technologies during the industrial era, and more recently, to anxieties surrounding pandemics and climate change.   
By analyzing movies’ emerging themes in the plot, the evolution of movies’ genre as well as the movies’ success (based on the IMDb rating), we can depict how the movie industry responded to the emergence and disappearance of major fears in the society. This analysis can be conducted on a global scale as well as on a regional scale.  
Examining the patterns of societal fears provides a deeper insight into the broader aspects of society. It reflects the historical, political and cultural context of the world across the years.

## Research Questions
1. What are the primary domains of fear explored in the database's movies, and how do they evolve chronologically? 
2. Do movies addressing current societal fears tend to have higher IMDb ratings compared to those exploring other themes? 
3. How has the number of movies addressing major global fears evolved since the dataset's inception? 
4. Are emerging fears covered in movies related to historical, political or cultural events?
5. What is the geographical distribution of the different fears addressed in international movies? 
6. What patterns emerge in the portrayal of fears in movies, and are there recurring combinations of fears frequently depicted on screen?
7. What evolution of the society can we depict from all of the previous results?


## Dataset
1. CMU Movie Summary Corpus. This dataset contains more than 42’000 movie plot summaries as well as general information about the movie such as the release date or the production country
2. IMDb: Incorporating the IMDb dataset into our analysis provides additional insights into the movies. Since there is a lot of missing values for the movies' box office revenues, will use the weighted average ratings data from title.ratings.tsv.gz to quantify a movie's success. IMDb is a recognised source for movie reviews. We therefore trust their methods for collecting and weighting this data. 
The IMDb dataset and our dataset use different identifiers for movies, IMDb employs "tconst," while our dataset uses the Freebase movie ID. In order to merge these datasets, we must establish a link between the two sets of IDs. We retrieve the correspondance from the Wikidata query service by performing a query in SPARSQL. Subsequently, we generate a correspondence table, removing any duplicate entries. With this completed correspondence table, we can proceed to merge the two datasets seamlessly.


## Files
'helpers.py' contains all functions that are used for preprocessing purposes.
'main.ipynb' is the main notebook containing all the code and results.
All preprocessed data can be found under 'data/MovieSummaries' folder.

## Methods 
* Pre-processing: to filter and arrange our data, we used the classic preprocessing methods.
* NLP topic detection: we used Natural Language Processing (NLP), and in particular the Latent Dirichlet Allocation (LDA). LDA is a generative statistical model used to classify text in a document to a particular topic. It builds a topic per document model and words per topic model, modeled as Dirichlet distributions. We used the LDA to perform topic detection on the plot summaries in order to point out particular fears depicted in the movies.
* Lexicons: to highlight the different fear categories, we create one lexicon per fear with the Empath() library.

## Proposed timeline 
1. Assess a list of fears that we want to extract from movies plot
2. Process the original data: clean, merge and display general interesting features of our data
3. Process the additional data: clean and merge it with the original data
4. Perform NLP on the plot to extract movies that treat those fears. 
5. Create a lexicon per fear categories, to chose which movies depict which fear.
6. Analyze the data from those movies, such as the IMDb average score or the crhonological and geographical distribution. Draw plots depicting interesting trends from our results
7. Conduct a precise analysis on war movies, based on their genre and their lexison. Draw plots depicting interesting trends from our results
8. Create the website, display all our interesting results and draw a conclusion of our analysis
 
## Team members contribution
* Faye: Analysis on fear categories, graphs and visualisation
* Colin: Optimisation, data cleaning, algorithm creation
* Romain: Lexicon creation, graphs and visualisation
* Clara: NLP topic detection, study case on war movies
* Octavio: Website building
Note: we often worked together on someone's computer. Thus, git contributions are not perfectly representative of the work done by each of the members.