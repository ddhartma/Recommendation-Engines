[image1]: assets/ident_neighbours.png "image1"
[image2]: assets/euc.png "image2"
[image3]: assets/sim_recom.png "image3"

# Recommendation Engines
Let's create Recommendation Engines for Movie Tweetings

## Outline
- [Recommendation Engines](#Recommendation_Engines)
  - [What's Ahead](#Whats_Ahead)
  - [Movie Tweeting Datasets](#Movie_Tweeting)
  - [Knowledge based recommendations](#Knowledge_based_recommendations)
  - [Collaborative Filtering based Recommendation](#Collaborative_Filtering_Based)
    - [Neighborhood Based Collaborative Filtering](#Neighborhood_Based_Collaborative_Filtering)
    - [Recommendations with Collaborative Filtering in Code](#Recom_with_Collab_Filter_in_Code)



- [Setup Instructions](#Setup_Instructions)
- [Acknowledgments](#Acknowledgments)
- [Further Links](#Further_Links)

# Recommendation Engines <a name="Recommendation_Engines"></a>

## What's Ahead <a name="Whats_Ahead"></a>
Types of Recommendations
In this lesson, you will be working with the MovieTweetings data to apply each of the three methods of recommendations:

- ***Knowledge Based*** Recommendations
- ***Collaborative Filtering Based*** Recommendations
- ***Content Based*** Recommendations

Within ***Collaborative Filtering***, there are two main branches:

- ***Model Based*** Collaborative Filtering
- ***Neighborhood Based*** Collaborative Filtering

***Similarity Metrics*** for Neighborhood Based Collaborative Filtering

Similarity between two users (or two items) including:

- Pearson's correlation coefficient
- Spearman's correlation coefficient
- Kendall's Tau
- Euclidean Distance
- Manhattan Distance

You will learn why sometimes one metric works better than another by looking at a specific situation where one metric provides more information than another.

***Business Cases For Recommendations***: 4 ideas to successfully implement recommendations to drive revenue, which include:

- Relevance
- Novelty
- Serendipity
- Increased Diversity

## Movie Tweeting Datasets <a name="Movie_Tweeting"></a>

Some Links:
 - [A Github account set up for MovieTweetings](https://github.com/sidooms/MovieTweetings)
 - [A slide deck by Simon Doom about MovieTweetings](https://www.slideshare.net/simondooms/rec-syschallenge2014intro?next_slideshow=1)

- Open notebook ```./notebooks/Introduction to the Recommendation Data.ipynb``` to check the ETL pipeline

## Knowledge based Recommendations <a name="Knowledge_based_recommendations"></a>

1. ***Rank Based Recommendations***:
    - Recommendation based on highest ratings, most purchases, most listened to, etc.
    - For example: Recommend most popular film of all film categories

2. ***Knowledge Based Recommendations***:
    - Knowledge about item or user preferences are used to make a recommendation
    - For example: Filter first for a certain film category (because user has chosen this category), then recommend most popular film out of this category

- Open notebook ```./notebooks/Most_Popular_Recommendations.ipynb```

  For this task, we will consider what is "most popular" based on the following criteria:

  - A movie with the highest average rating is considered best
  - With ties, movies that have more ratings are better
  - A movie must have a minimum of 5 ratings to be considered among the best movies
  - If movies are tied in their average rating and number of ratings, the ranking is determined by the movie that is the most recent rating

  With these criteria, the goal for this notebook is to take a user_id and provide back the n_top recommendations. Use the function below as the scaffolding that will be used for all the future recommendations as well.

    ```
    def create_ranked_df(movies, reviews):
        '''
        INPUT
        movies - the movies dataframe
        reviews - the reviews dataframe

        OUTPUT
        ranked_movies - a dataframe with movies that are sorted by highest avg rating, more reviews,
                        then time, and must have more than 4 ratings
        '''

        # Pull the average ratings and number of ratings for each movie
        movie_ratings = reviews.groupby('movie_id')['rating']
        avg_ratings = movie_ratings.mean()
        num_ratings = movie_ratings.count()
        last_rating = pd.DataFrame(reviews.groupby('movie_id').max()['date'])
        last_rating.columns = ['last_rating']

        # Add Dates
        rating_count_df = pd.DataFrame({'avg_rating': avg_ratings, 'num_ratings': num_ratings})
        rating_count_df = rating_count_df.join(last_rating)

        # merge with the movies dataset
        movie_recs = movies.set_index('movie_id').join(rating_count_df)

        # sort by top avg rating and number of ratings
        ranked_movies = movie_recs.sort_values(['avg_rating', 'num_ratings', 'last_rating'], ascending=False)

        # for edge cases - subset the movie list to those with only 5 or more reviews
        ranked_movies = ranked_movies[ranked_movies['num_ratings'] > 4]

        return ranked_movies


  def popular_recommendations(user_id, n_top, ranked_movies):
        '''
        INPUT:
        user_id - the user_id (str) of the individual you are making recommendations for
        n_top - an integer of the number recommendations you want back
        ranked_movies - a pandas dataframe of the already ranked movies based on avg rating, count, and time

        OUTPUT:
        top_movies - a list of the n_top recommended movies by movie title in order best to worst
        '''

        top_movies = list(ranked_movies['movie'][:n_top])

        return top_movies


  def popular_recs_filtered(user_id, n_top, ranked_movies, years=None, genres=None):
        '''
        REDO THIS DOC STRING

        INPUT:
        user_id - the user_id (str) of the individual you are making recommendations for
        n_top - an integer of the number recommendations you want back
        ranked_movies - a pandas dataframe of the already ranked movies based on avg rating, count, and time
        years - a list of strings with years of movies
        genres - a list of strings with genres of movies

        OUTPUT:
        top_movies - a list of the n_top recommended movies by movie title in order best to worst
        '''
        # Filter movies based on year and genre
        if years is not None:
            ranked_movies = ranked_movies[ranked_movies['date'].isin(years)]

        if genres is not None:
            num_genre_match = ranked_movies[genres].sum(axis=1)
            ranked_movies = ranked_movies.loc[num_genre_match > 0, :]


        # create top movies list
        top_movies = list(ranked_movies['movie'][:n_top])

        return top_movies
  ```

## Collaborative Filtering based Recommendation <a name="Collaborative_Filtering_Based"></a>
- Collaborative filtering has two senses, a ***narrow*** one and a more ***general*** one:
    - In the newer, ***narrower*** sense, collaborative filtering is a method of making ***automatic predictions*** (filtering) about the ***interests of a user*** by collecting preferences or taste ***information from many users*** (collaborating)
    - In the more ***general*** sense, collaborative filtering is the ***process of filtering for information*** or patterns using techniques involving collaboration among ***multiple agents, viewpoints, data sources***, etc.
- A method of making recommendations based on ***using the collaboration of user-item interactions***
- Even without background information of the user and the items one can make still recommendations
- We need only: ***Information about how users and items interact with one another***
- Examples of Data Collaborative Filtering:
    - Item ratings for each user
    - Item liked by user or not
    - Item used by user or not

Within ***Collaborative Filtering***, there are two main branches:

- ***Model Based*** Collaborative Filtering
- ***Neighborhood Based*** Collaborative Filtering

## Neighborhood Based Collaborative Filtering <a name="Neighborhood_Based_Collaborative_Filtering"></a>
- We are interested in finding individuals that are closely related to one another

    ![image1]

- Open notebook ```./notebooks/Measuring_Simularity.ipynb```

- ***Pearson's Correlation***
    - Statistical relationshipetween two continuous variables.  
    - It is known as the best method of measuring the association between variables
    - It is based on the method of covariance.  
    - It gives information about the magnitude of correlation, as well as the direction of the relationship

    - <img src="https://render.githubusercontent.com/render/math?math=CORR(x, y) = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i-\bar{x})^2}\sqrt{\sum_{i=1}^{n}(y_i-\bar{y})^2}} " height="80px">

    where 

    - <img src="https://render.githubusercontent.com/render/math?math=\bar{x} = \frac{1}{n}\sum_{i=1}^{n}x_i" height="40px">

    ```
    def pearson_corr(x, y):
        '''
        INPUT
        x - an array of matching length to array y
        y - an array of matching length to array x
        OUTPUT
        corr - the pearson correlation coefficient for comparing x and y
        '''

        # Compute Mean Values
        mean_x, mean_y = np.sum(x)/len(x), np.sum(y)/len(y) 
        
        x_diffs = x - mean_x
        y_diffs = y - mean_y
        numerator = np.sum(x_diffs*y_diffs)
        denominator = np.sqrt(np.sum(x_diffs**2))*np.sqrt(np.sum(y_diffs**2))
            
        corr = numerator/denominator
        
        return corr                    
    ```
- ***Spearman's Correlation***

    - The Spearman correlation is a nonparametric measure of rank correlation 
    - The Spearman correlation between two variables is equal to the Pearson correlation between the rank values of those two variables; 
    - while Pearson's correlation assesses linear relationships, Spearman's correlation assesses monotonic relationships (whether linear or not). 
    - If there are no repeated data values, a perfect Spearman correlation of +1 or −1 occurs when each of the variables is a perfect monotone function of the other.
    - Intuitively, the Spearman correlation between two variables will be high when observations have a similar (or identical for a correlation of 1) rank (i.e. relative position label of the observations within the variable: 1st, 2nd, 3rd, etc.) between the two variables, and low when observations have a dissimilar (or fully opposed for a correlation of −1) rank between the two variables. 

    Transform input vectors to ranked vectors via the ```rank()``` method
    - The ranked values for the variable x1 are: [ 1.  2.  3.  4.  5.  6.  7.]
    - The raw data values for the variable x1 are: [-3 -2 -1  0  1  2  3]

    - <img src="https://render.githubusercontent.com/render/math?math=x \rightarrow x^{r}" height="20px">

    - <img src="https://render.githubusercontent.com/render/math?math=y \rightarrow y^{r}" height="25px">

    - <img src="https://render.githubusercontent.com/render/math?math=CORR(x, y) = \frac{\sum_{i=1}^{n}(x^{r}_i - \bar{x}^{r})(y^{r}_i - \bar{y}^{r})}{\sqrt{\sum_{i=1}^{n}(x^{r}_i-\bar{x}^{r})^2}\sqrt{\sum_{i=1}^{n}(y^{r}_i-\bar{y}^{r})^2}}" height="90px">

    ```
    def corr_spearman(x, y):
        '''
        INPUT
        x - an array of matching length to array y
        y - an array of matching length to array x
        OUTPUT
        corr - the spearman correlation coefficient for comparing x and y
        '''
        
        # Compute Mean Values
        x = x.rank()
        y = y.rank()
        mean_x, mean_y = np.sum(x)/len(x), np.sum(y)/len(y) 
        
        x_diffs = x - mean_x
        y_diffs = y - mean_y
        numerator = np.sum(x_diffs*y_diffs)
        denominator = np.sqrt(np.sum(x_diffs**2))*np.sqrt(np.sum(y_diffs**2))
            
        corr = numerator/denominator
                    
        return corr  
    ```

- ***Kendall's Tau***

    - Kendall's Tau is a nonparametric measure of rank correlation 
    - Kendall's Tau is a statistic used to measure the ordinal association between two measured quantities
    - Intuitively, the Kendall correlation between two variables will be high when observations have a similar (or identical for a correlation of 1) rank (i.e. relative position label of the observations within the variable: 1st, 2nd, 3rd, etc.) between the two variables, and low when observations have a dissimilar (or fully different for a correlation of −1) rank between the two variables. 
    
    Transform input vectors to ranked vectors via the ```rank()``` method
    - The ranked values for the variable x1 are: [ 1.  2.  3.  4.  5.  6.  7.]
    - The raw data values for the variable x1 are: [-3 -2 -1  0  1  2  3]

    - <img src="https://render.githubusercontent.com/render/math?math=x \rightarrow x^{r}" height="20px">

    - <img src="https://render.githubusercontent.com/render/math?math=y \rightarrow y^{r}" height="25px">

    - <img src="https://render.githubusercontent.com/render/math?math=TAU(x, y) = \frac{2}{n(n -1)}\sum_{i < j}sgn(x^r_i - x^r_j)sgn(y^r_i - y^r_j)" height="50px">


    - <img src="https://render.githubusercontent.com/render/math?math=sgn(x^r_i - x^r_j)sgn(y^r_i - y^r_j)" height="40px">

    - possible results for sgn(...) * sgn(...) are -1, 0, 1

    ```
    def kendalls_tau(x, y):
        '''
        INPUT
        x - an array of matching length to array y
        y - an array of matching length to array x
        OUTPUT
        tau - the kendall's tau for comparing x and y
        '''  

        # Change each vector to ranked values
        x = x.rank()
        y = y.rank()
        n = len(x)
        
        sum_vals = 0
        # Compute Mean Values
        for i, (x_i, y_i) in enumerate(zip(x, y)):
            for j, (x_j, y_j) in enumerate(zip(x, y)):
                if i < j:
                    sum_vals += np.sign(x_i - x_j)*np.sign(y_i - y_j)
                            
        tau = 2*sum_vals/(n*(n-1))
        
        return tau
    ```

- ***Euclidian Distance***

    - Euclidean distance can also just be considered as straight-line distance between two vectors.
    - It is the Pythagorean distance

    - <img src="https://render.githubusercontent.com/render/math?math=EUC(x, y) = \sqrt{\sum_{i=1}^{n}(x_i - y_i)^2}" height="40px">
    

    ```
    def eucl_dist(x, y):
        '''
        INPUT
        x - an array of matching length to array y
        y - an array of matching length to array x
        OUTPUT
        euc - the euclidean distance between x and y
        '''  
        return np.linalg.norm(x - y)
    ```

- ***Manhattan Distance***

    - The distance between two points measured along axes at right angles
    - Different from euclidean distance, Manhattan distance is a 'manhattan block' distance from one vector to another. Therefore, you can imagine this distance as a way to compute the distance between two points when you are not able to go through buildings.
    

    - <img src="https://render.githubusercontent.com/render/math?math=MANHATTAN(x, y) = \sqrt{\sum_{i=1}^{n}|x_i - y_i|}" height="40px">

    ![image2]

    You can see in the above image, the ***blue*** line gives the ***Manhattan*** distance, while the ***green*** line gives the ***Euclidean*** distance between two points.

    ```
    def manhat_dist(x, y):
        '''
        INPUT
        x - an array of matching length to array y
        y - an array of matching length to array x
        OUTPUT
        manhat - the manhattan distance between x and y
        '''  
        
        return sum(abs(e - s) for s, e in zip(x, y))
    ```

    ![image3]
## Recommendations with Collaborative Filtering in Code <a name="Recom_with_Collab_Filter_in_Code"></a>

- Open notebook ```./notebooks/Collaborative Filtering.ipynb```
- User-User Based Collaborative Filtering

    ```
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import tests as t
    from scipy.sparse import csr_matrix
    from IPython.display import HTML

    %matplotlib inline

    # Read in the datasets
    movies = pd.read_csv('movies_clean.csv')
    reviews = pd.read_csv('reviews_clean.csv')

    del movies['Unnamed: 0']
    del reviews['Unnamed: 0']
    ```
    ```
    user_items = reviews[['user_id', 'movie_id', 'rating']]
    user_items.head()
    ```

    - Let's create a user-item matrix, the base for the Recommendation system
    - You can do this via a [Pivot Table](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.pivot_table.html)
    - Better (less memory consuming) [groupby method](https://stackoverflow.com/questions/39648991/pandas-dataframe-pivot-not-fitting-in-memory)
    ```
    # Create user-by-item matrix
    user_by_movie = user_items.groupby(['user_id', 'movie_id'])['rating'].max().unstack()
    ```
    ```
    # Create a dictionary with users and corresponding movies seen
    def movies_watched(user_id):
        '''
        INPUT:
        user_id - the user_id of an individual as int
        OUTPUT:
        movies - an array of movies the user has watched
        '''
        movies = user_by_movie.loc[user_id][user_by_movie.loc[user_id].isnull() == False].index.values

        return movies

    def create_user_movie_dict():
        '''
        INPUT: None
        OUTPUT: movies_seen - a dictionary where each key is a user_id and the value is an array of movie_ids
        
        Creates the movies_seen dictionary
        '''
        n_users = user_by_movie.shape[0]
        movies_seen = dict()

        for user1 in range(1, n_users+1):
            
            # assign list of movies to each user key
            movies_seen[user1] = movies_watched(user1)
        
        return movies_seen
        
    movies_seen = create_user_movie_dict()
    ```

    ```
    # Remove individuals who have watched 2 or fewer movies - don't have enough data to make recs
    def create_movies_to_analyze(movies_seen, lower_bound=2):
        '''
        INPUT:  
        movies_seen - a dictionary where each key is a user_id and the value is an array of movie_ids
        lower_bound - (an int) a user must have more movies seen than the lower bound to be added to the movies_to_analyze dictionary

        OUTPUT: 
        movies_to_analyze - a dictionary where each key is a user_id and the value is an array of movie_ids
        
        The movies_seen and movies_to_analyze dictionaries should be the same except that the output dictionary has removed 
        
        '''
        movies_to_analyze = dict()

        for user, movies in movies_seen.items():
            if len(movies) > lower_bound:
                movies_to_analyze[user] = movies
        return movies_to_analyze

    movies_to_analyze = create_movies_to_analyze(movies_seen)
    ```
    ```
    # compute correlations
    def compute_correlation(user1, user2):
        '''
        INPUT
        user1 - int user_id
        user2 - int user_id
        OUTPUT
        the correlation between the matching ratings between the two users
        '''
        # Pull movies for each user
        movies1 = movies_to_analyze[user1]
        movies2 = movies_to_analyze[user2]
        
        
        # Find Similar Movies
        sim_movs = np.intersect1d(movies1, movies2, assume_unique=True)
        
        # Calculate correlation between the users
        df = user_by_movie.loc[(user1, user2), sim_movs]
        corr = df.transpose().corr().iloc[0,1]
        
        return corr #return the correlation
    ```
    ```
    def compute_euclidean_dist(user1, user2):
        '''
        INPUT
        user1 - int user_id
        user2 - int user_id
        OUTPUT
        the euclidean distance between user1 and user2
        '''
        # Pull movies for each user
        movies1 = movies_to_analyze[user1]
        movies2 = movies_to_analyze[user2]
        
        
        # Find Similar Movies
        sim_movs = np.intersect1d(movies1, movies2, assume_unique=True)
        
        # Calculate euclidean distance between the users
        df = user_by_movie.loc[(user1, user2), sim_movs]
        dist = np.linalg.norm(df.loc[user1] - df.loc[user2])
        
        return dist #return the euclidean distance
    ```

    ```
    # Read in solution euclidean distances"
    import pickle
    df_dists = pd.read_pickle("data/Term2/recommendations/lesson1/data/dists.p")
    ```

    ```
    def find_closest_neighbors(user):
        '''
        INPUT:
            user - (int) the user_id of the individual you want to find the closest users
        OUTPUT:
            closest_neighbors - an array of the id's of the users sorted from closest to farthest away
        '''
        # I treated ties as arbitrary and just kept whichever was easiest to keep using the head method
        # You might choose to do something less hand wavy
        
        closest_users = df_dists[df_dists['user1']==user].sort_values(by='eucl_dist').iloc[1:]['user2']
        closest_neighbors = np.array(closest_users)
        
        return closest_neighbors
        
        
    def find_closest_neighbors(user):
        '''
        INPUT:
            user - (int) the user_id of the individual you want to find the closest users
        OUTPUT:
            closest_neighbors - an array of the id's of the users sorted from closest to farthest away
        '''
        # I treated ties as arbitrary and just kept whichever was easiest to keep using the head method
        # You might choose to do something less hand wavy
        
        closest_users = df_dists[df_dists['user1']==user].sort_values(by='eucl_dist').iloc[1:]['user2']
        closest_neighbors = np.array(closest_users)
        
        return closest_neighbors
        
        
    def movies_liked(user_id, min_rating=7):
        '''
        INPUT:
        user_id - the user_id of an individual as int
        min_rating - the minimum rating considered while still a movie is still a "like" and not a "dislike"
        OUTPUT:
        movies_liked - an array of movies the user has watched and liked
        '''
        movies_liked = np.array(user_items.query('user_id == @user_id and rating > (@min_rating -1)')['movie_id'])
        
        return movies_liked


    def movie_names(movie_ids):
        '''
        INPUT
        movie_ids - a list of movie_ids
        OUTPUT
        movies - a list of movie names associated with the movie_ids
        
        '''
        movie_lst = list(movies[movies['movie_id'].isin(movie_ids)]['movie'])
    
        return movie_lst
        

    def make_recommendations(user, num_recs=10):
        '''
        INPUT:
            user - (int) a user_id of the individual you want to make recommendations for
            num_recs - (int) number of movies to return
        OUTPUT:
            recommendations - a list of movies - if there are "num_recs" recommendations return this many
                            otherwise return the total number of recommendations available for the "user"
                            which may just be an empty list
        '''
        # I wanted to make recommendations by pulling different movies than the user has already seen
        # Go in order from closest to farthest to find movies you would recommend
        # I also only considered movies where the closest user rated the movie as a 9 or 10
        
        # movies_seen by user (we don't want to recommend these)
        movies_seen = movies_watched(user)
        closest_neighbors = find_closest_neighbors(user)
        
        # Keep the recommended movies here
        recs = np.array([])
        
        # Go through the neighbors and identify movies they like the user hasn't seen
        for neighbor in closest_neighbors:
            neighbs_likes = movies_liked(neighbor)
            
            #Obtain recommendations for each neighbor
            new_recs = np.setdiff1d(neighbs_likes, movies_seen, assume_unique=True)
            
            # Update recs with new recs
            recs = np.unique(np.concatenate([new_recs, recs], axis=0))
            
            # If we have enough recommendations exit the loop
            if len(recs) > num_recs-1:
                break
        
        # Pull movie titles using movie ids
        recommendations = movie_names(recs)
        
        return recommendations


    def all_recommendations(num_recs=10):
        '''
        INPUT 
            num_recs (int) the (max) number of recommendations for each user
        OUTPUT
            all_recs - a dictionary where each key is a user_id and the value is an array of recommended movie titles
        '''
        
        # All the users we need to make recommendations for
        users = np.unique(df_dists['user1'])
        n_users = len(users)
        
        #Store all recommendations in this dictionary
        all_recs = dict()
        
        # Make the recommendations for each user
        for user in users:
            all_recs[user] = make_recommendations(user, num_recs)
        
        return all_recs

    all_recs = all_recommendations(10)
    ```


## Setup Instructions <a name="Setup_Instructions"></a>
The following is a brief set of instructions on setting up a cloned repository.

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites: Installation of Python via Anaconda and Command Line Interaface <a name="Prerequisites"></a>
- Install [Anaconda](https://www.anaconda.com/distribution/). Install Python 3.7 - 64 Bit
- If you need a Command Line Interface (CLI) under Windows you could use [git](https://git-scm.com/). Under Mac OS use the pre-installed Terminal.

- Upgrade Anaconda via
```
$ conda upgrade conda
$ conda upgrade --all
```

- Optional: In case of trouble add Anaconda to your system path. Write in your CLI
```
$ export PATH="/path/to/anaconda/bin:$PATH"
```

### Clone the project <a name="Clone_the_project"></a>
- Open your Command Line Interface
- Change Directory to your project older, e.g. `cd my_github_projects`
- Clone the Github Project inside this folder with Git Bash (Terminal) via:
```
$ git clone https://github.com/ddhartma/Recommendation-Engines.git
```

- Change Directory
```
$ cd Recommendation-Engines
```

- Create a new Python environment, e.g. rec_eng. Inside Git Bash (Terminal) write:
```
$ conda create --name rec_eng
```

- Activate the installed environment via
```
$ conda activate rec_eng
```

- Install the following packages (via pip or conda)
```
numpy = 1.17.4
pandas = 0.24.2
```

- Check the environment installation via
```
$ conda env list
```

## Acknowledgments <a name="Acknowledgments"></a>
* This project is part of the Udacity Nanodegree program 'Data Science'. Please check this [link](https://www.udacity.com) for more information.

## Further Links <a name="Further_Links"></a>

Git/Github
* [GitFlow](https://datasift.github.io/gitflow/IntroducingGitFlow.html)
* [A successful Git branching model](https://nvie.com/posts/a-successful-git-branching-model/)
* [5 types of Git workflows](https://buddy.works/blog/5-types-of-git-workflows)

Docstrings, DRY, PEP8
* [Python Docstrings](https://www.geeksforgeeks.org/python-docstrings/)
* [DRY](https://www.youtube.com/watch?v=IGH4-ZhfVDk)
* [PEP 8 -- Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/)
