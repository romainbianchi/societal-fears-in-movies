import pandas as pd
import numpy as np
import datetime
import requests
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('averaged_perceptron_tagger')
from sklearn.metrics.pairwise import cosine_similarity
from skfda.inference.hotelling import hotelling_t2
from skfda.representation.grid import FDataGrid
import pycountry_convert as pc


fear_categories = ['war','climate change', 'terrorism', 'pandemic', 'economic collapse', 'technology', 'alien']



def valid_format(date_string, date_format='%Y-%m-%d'):
    """
    check if the string in input is in the given format

    params: 
        date_string: string representing a date
        date_format: format of the date string for comparison

    return: 
        True if the date string is in the format YYYY-MM-DD
        False otherwise
    """

    try:
        datetime.datetime.strptime(date_string, date_format)
        return True
    except ValueError:
        return False
    

    

def keep_the_year(date_full, key):
    """
    convert a column containing date with different format keeping only the year using the pandas apply function
    
    params: 
        date_full: a dataframe containing a column with date in different format
        key: the name of the column containing the date
        
    return:
        a dataframe containing only the year of each date, in place of the original column
    """

    def keep_the_year_apply_helper(date_string):
        """
        helper function for keep_the_year_apply
        """
        # Define date formats regarding the three different formats present in the dataset
        format1 = '%Y-%m-%d'
        format2 = '%Y-%m'
        format3 = '%Y'

        # If the date is out of bounds, consider it as missing value and continue
        if str(date_string) > '2023' or str(date_string) < '1800' or str(date_string) == ' ' or str(date_string) == 'nan':    # Even with different date formats the inequality works
            return np.nan

        if valid_format(date_string, format1):
            return datetime.datetime.strptime(date_string, format1).date().year
        elif valid_format(date_string, format2):
            return datetime.datetime.strptime(date_string, format2).date().year
        elif valid_format(date_string, format3):
            return datetime.datetime.strptime(date_string, format3).date().year
        else: 
            return np.nan
        
    # Apply the helper function to the column
    date_full[key] = date_full[key].apply(keep_the_year_apply_helper).astype('Int64')

    return date_full




def link_tconst_freebaseID():
    """
    Wikidata query to get the link between IMDb tconst and freebaseID

    params:
        -

    return: 
        a dataframe containing IMDb tconst and the corresponding freebase ID of the a movie
    """

    # Wikidata SPARQL endpoint
    url = 'https://query.wikidata.org/sparql'

    # Query to get freebase ID and IMDb ID
    # wdt:P345 IMDb ID in wikidata
    # wdt:P646 Freebase ID in wikidata
    query = """
    SELECT ?item ?tconst ?freebaseID WHERE {
        ?item wdt:P345 ?tconst.
        OPTIONAL {?item wdt:P646 ?freebaseID}
    }
    """

    # Query
    params = {'query': query, 'format': 'json'}
    data = requests.get(url ,params = params).json()

    # Create a dataframe that link IMDb tconst and freebaseID
    tconst = []
    freebase_id = []
    for item in data['results']['bindings']:
        tconst.append(item['tconst']['value'])
        freebase_id_val = item.get('freebaseID', {}).get('value', np.nan)
        freebase_id.append(freebase_id_val)

    return pd.DataFrame(data={'tconst': tconst, 'Freebase movie ID': freebase_id})



def get_cleaned_data(path):
    """
    Get the data from the given path and clean it

    params:
        path: the path of the folder containing the data

    return:
        the cleaned datasets
    """
    print("Loading the data...")
    # Load data/moviesummaries/plot_summaries.txt
    plot_summaries = get_summaries(path)

    # Load data/moviesummaries/movie.metadata.tsv
    movie_metadata = pd.read_csv(path + 'moviesummaries/movie.metadata.tsv', sep='\t', header=None)
    movie_metadata.columns = ["Wikipedia movie ID", "Freebase movie ID", "Movie name", "Movie release date", "Movie revenue", "Movie runtime",
                            "Movie languages", "Movie countries", "Movie genres"]

    print("Cleaning the data...")
    # Merge 'left' the movie_metadata and plot_summaries dataframes on the Wikipedia movie ID column
    all_movies = movie_metadata.merge(plot_summaries, on="Wikipedia movie ID", how="left")

    # Drop one of each pair of duplicates
    all_movies.drop_duplicates(subset=["Movie name", "Movie release date", "Movie revenue", "Movie languages", "Movie genres", "Movie countries", "Movie runtime", "Summary"], inplace=True, keep="first")
    
    # Converting the movie release date to keep only the year for the all_movie table
    all_movies = keep_the_year(all_movies, key='Movie release date')

    # Some columns contains dicts. Let's only keep the values of these dicts as lists since we don't care about their keys
    all_movies['Movie genres'] = [list(eval(genre).values()) for genre in all_movies['Movie genres']]
    all_movies['Movie languages'] = [list(eval(genre).values()) for genre in all_movies['Movie languages']]
    all_movies['Movie countries'] = [list(eval(genre).values()) for genre in all_movies['Movie countries']]


    print("Adding IMDb ratings...")
    # Add IMDb ratings
    movie_ratings = pd.read_csv(path + 'title.ratings.tsv', sep='\t', header=0)

    # Create the table
    link_id = link_tconst_freebaseID()

    # Drop duplicates
    link_id = link_id.drop_duplicates(subset=['tconst'])
    link_id = link_id.drop_duplicates(subset=['Freebase movie ID'])

    # Add freebase ID to movie_ratings
    movie_ratings = pd.merge(movie_ratings, link_id, on='tconst', how='left')

    # Merge all_movies and movie_ratings
    all_movies = pd.merge(all_movies, movie_ratings, on='Freebase movie ID', how='left')
    # Drop tconst column
    all_movies.drop(columns=['tconst'], inplace=True)

    return all_movies



def get_summaries(path, punctuation=True, casefolding = True, stop_words=True, lemmatize=True, movie_film=True, remove_names = True, force_reload=False, save=True):
    '''
    Get the summaries from the given path and clean them
    
    params:
        path: the path of the folder containing the data
        punctuation: boolean to remove punctuation
        casefolding: boolean to apply casefolding
        stop_words: boolean to remove stop words
        lemmatize: boolean to lemmatize
        movie_film: boolean to remove the words film and films
        remove_names: boolean to remove the most common names
        force_reload: boolean to force the reload of the processed summaries
        save: boolean to save the processed summaries
        
    return:
        the cleaned summaries
    '''
    
    print("Loading and cleaning the summaries...")

    # Dataset downloaded from: https://data.world/davidam/international-names/workspace/data-dictionary 
    names = pd.read_csv(path + 'moviesummaries/interall.csv')
    array_names = names.iloc[:,0].dropna().tolist()
    array_names = [s.lower() for s in array_names]

    # Check if processed_summaries.tsv exists
    try:
        if force_reload:
            raise FileNotFoundError
        plot_summaries = pd.read_csv(path + 'moviesummaries/processed_summaries.tsv', sep='\t', header=0)
        print("Summaries loaded from processed_summaries.tsv")
        return plot_summaries
    except FileNotFoundError:
        print("processed_summaries.tsv not found, processing the summaries...")

    # Load data/moviesummaries/plot_summaries.txt
    plot_summaries = pd.read_csv(path + 'moviesummaries/plot_summaries.txt', sep='\t', header=None)
    plot_summaries.columns = ["Wikipedia movie ID", "Summary"]

    # Tokenize
    print("Tokenizing...")
    plot_summaries['Summary'] = plot_summaries['Summary'].apply(lambda x: word_tokenize(x))

    # Remove punctuation
    if punctuation:
        print("Removing punctuation...")
        plot_summaries['Summary'] = plot_summaries['Summary'].apply(lambda x: [word for word in x if word.isalpha()])

    # Casefolding
    if casefolding:
        print("Casefolding...")
        plot_summaries['Summary'] = plot_summaries['Summary'].apply(lambda x: [word.lower() for word in x])

    # Remove stop words and common words
    if stop_words:
        print("Removing stop words and common words/names...")
        stop_words = set(stopwords.words('english'))
        if movie_film:
            print("Removing common words...")
            stop_words.update(['film', 'films', 'movie', 'movies'])        
        if remove_names:
            print("Removing common names...")
            stop_words.update(array_names)
        plot_summaries['Summary'] = plot_summaries['Summary'].apply(lambda x: [word for word in x if word.lower() not in stop_words])

    # Lemmatize
    if lemmatize:
        print("Lemmatizing...")
        lemmatizer = nltk.stem.WordNetLemmatizer()
        plot_summaries['Summary'] = plot_summaries['Summary'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

    # Join
    plot_summaries['Summary'] = plot_summaries['Summary'].apply(lambda x: ' '.join(x))

    # Save
    if save:
        print("Saving the processed summaries...")
        plot_summaries.to_csv(path + 'moviesummaries/processed_summaries.tsv', sep='\t', index=False)

    return plot_summaries


# top k function that returns the top k words for each topic
def top_k(data, category, k):
    return data.sort_values(by=category, ascending=False).head(k)

# top k function that returns the top k words for all topics as a list
def top_k_all(data, categories, k):
    top_k_all = []
    for category in categories:
        top_k_all.append(top_k(data, category, k))
    return top_k_all

# top threshold function that returns the documents with a frequency above a threshold for each topic
def top_threshold(data, category, threshold):
    return data[data[category] >= threshold].sort_values(by=category, ascending=False)

# top threshold function that returns the documents with a frequency above a threshold for all topics as a list
def top_threshold_all(data, categories, threshold):
    top_threshold_all = []
    for category in categories:
        top_threshold_all.append(top_threshold(data, category, threshold))
    return top_threshold_all

# top p function that returns the top p % documents for each topic
def top_p(data, category, p):
    return data[data[category] >= data[category].quantile(p)].sort_values(by=category, ascending=False)

# top p function that returns the top p % documents for all topics as a list
def top_p_all(data, categories, p):
    top_p_all = []
    for category in categories:
        top_p_all.append(top_p(data, category, p))
    return top_p_all

# functions to compute similarity between two sets of movies, based on their lexicon values
# cosine similarity based function: since the lexicon values are positive, the cosine similarity return values between 0 and 1
def cos_similarity(movie_set_1, movie_set_2, lexicon_columns = fear_categories):
    
    # Extract the relevant columns for the sets
    set_1_data = movie_set_1[lexicon_columns].values
    set_2_data = movie_set_2[lexicon_columns].values

    # Calculate the cosine similarity
    similarity_matrix = cosine_similarity(set_1_data, set_2_data)

    # Calculate the overall similarity as the mean of all similarities
    overall_similarity = similarity_matrix.mean()

    return overall_similarity

# euclidean distance based function
def euclidean_similarity(movie_set_1, movie_set_2, lexicon_columns = fear_categories):
    
    # Extract the relevant columns for the sets
    set_1_data = movie_set_1[lexicon_columns].values
    set_2_data = movie_set_2[lexicon_columns].values

    # Calculate the euclidean distance
    distance_matrix = np.linalg.norm(set_1_data - set_2_data, axis=1)

    # Calculate the overall distance as the mean of all distances
    overall_distance = distance_matrix.mean()

    return -overall_distance

# Hotelling's T2 based function
def hotelling_similarity(movie_set_1, movie_set_2, lexicon_columns = fear_categories):
    
    # Extract the relevant columns for the sets
    set_1_data = movie_set_1[lexicon_columns].values
    set_2_data = movie_set_2[lexicon_columns].values

    # put the data as FData
    set_1_data = FDataGrid(set_1_data)
    set_2_data = FDataGrid(set_2_data)

    # Calculate the Hotelling's T2
    t2 = hotelling_t2(set_1_data, set_2_data)

    return -t2

# function to optimize the number of top documents to keep for each category with respect to the similarity to a random set of movies
def top_optimize(data, category, similarity, lower, upper, agg, r=10, seed=42):
    """Optimize the number of top documents to keep for a given category with respect to the similarity to a random set of movies

    Args:
        data (DataFrame): DataFrame containing the documents
        category (str): category to optimize
        similarity (function): similarity function to use
        lower (int): lower bound of the number of documents to keep
        upper (int): upper bound of the number of documents to keep
        agg (function): aggregation function to use on the random graphs similarities
        r (int, optional): number of random sets of movies to use for similarity. Defaults to 10.
        seed (int, optional): seed to use for the random sets of movies. Defaults to 42.

    Returns:
        DataFrame: DataFrame containing the top documents
        list: list of similarities for each number of documents
    """
    sorted_data = data.sort_values(by=category, ascending=False)
    similarities = []
    for k in range(lower, upper):
        # get the top k documents
        top_k = sorted_data.head(k)
        
        # Create r random sets of movies of size k
        random_sets = []
        for i in range(r):
            random_sets.append(data.sample(k, random_state=seed + i))

        # compute the similarity between the top k documents and each random set of movies
        random_sets_similarities = []
        for random_set in random_sets:
            random_sets_similarities.append(similarity(top_k, random_set))

        # compute the similarity between the top k documents and the r random sets of movies together
        similarities.append(agg(random_sets_similarities))

    # get the index of the lowest similarity
    index = similarities.index(min(similarities))
    
    return sorted_data.head(lower + index), similarities

# function to optimize the number of top documents to keep over all categories with respect to the similarity to a random set of movies
def top_optimize_all(data, categories, similarity, lower, upper, r_agg, cat_agg, r=10, seed=42):
    """Optimize the number of top documents to keep over all categories with respect to the similarity to a random set of movies

    Args:
        data (DataFrame): DataFrame containing the documents
        categories (list): list of categories to optimize
        similarity (function): similarity function to use
        lower (int): lower bound of the number of documents to keep
        upper (int): upper bound of the number of documents to keep
        r_agg (function): aggregation function to use on the random graphs similarities
        cat_agg (function): aggregation function to use on the categories similarities
        r (int, optional): number of random sets of movies to use for similarity. Defaults to 10.
        seed (int, optional): seed to use for the random sets of movies. Defaults to 42.

    Returns:
        DataFrame: DataFrame containing the top documents
        list: list of similarities for each number of documents
    """
    sorted_data = top_k_all(data, categories, upper)
    similarities = []
    for k in range(lower, upper):
        # Create r random sets of movies of size k
        random_sets = []
        for i in range(r):
            random_sets.append(data.sample(k, random_state=seed + i))

        categories_similarities = []

        for category in categories:
            # get the top k documents for each category
            top_k = sorted_data[categories.index(category)].head(k)

            # compute the similarity between the top k documents and each random set of movies
            random_sets_similarities = []
            for random_set in random_sets:
                random_sets_similarities.append(similarity(top_k, random_set, categories))

            # compute the similarity between the top k documents and the r random sets of movies together
            categories_similarities.append(r_agg(random_sets_similarities))

        # aggregate categories similarities
        similarities.append(cat_agg(categories_similarities))

    # get the index of the lowest similarity
    index = similarities.index(min(similarities))

    return top_k_all(data, categories, lower + index), similarities

# max, min, mean, median, sum aggregate functions
def max_agg(data):
    return max(data)

def min_agg(data):
    return min(data)

def mean_agg(data):
    return np.mean(data)

def median_agg(data):
    return np.median(data)

def sum_agg(data):
    return sum(data)


def country_to_continent(country):
    """
    # Return the continent of a country

    params: 
        country: the name of the country
    return:
        the continent of the country
    """


    if country not in pc.map_countries().keys():
        return np.nan
    
    pc.country_name_to_country_alpha2(country)
    continent_code = pc.country_alpha2_to_continent_code(pc.country_name_to_country_alpha2(country))
    continent_name = pc.convert_continent_code_to_continent_name(continent_code)
    return continent_name