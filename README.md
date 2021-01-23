[image1]: assets/ident_neighbours.png "image1"
[image2]: assets/euc.png "image2"
[image3]: assets/sim_recom.png "image3"
[image4]: assets/content_based_recom.png "image4"
[image5]: assets/dot_sim.png "image5"
[image6]: assets/loc_based_recom.png "image6"
[image7]: assets/deep_learn_recom.png "image7"
[image8]: assets/type_rating.png "image8"

# Recommendation Engines 
Let's create Recommendation Engines for Movie Tweetings.
This is the base of how e.g. Facebook offers personalized advertising to you. You will learn in this session how Recommendation Engines make predictions on your preferences based on your historical behavior.


## Outline
- [Recommendation Engines](#Recommendation_Engines)
  - [What's Ahead](#Whats_Ahead)
  - [Movie Tweeting Datasets](#Movie_Tweeting)
  - [Knowledge based recommendations](#Knowledge_based_recommendations)
  - [Collaborative Filtering based Recommendation](#Collaborative_Filtering_Based)
    - [Neighborhood Based Collaborative Filtering](#Neighborhood_Based_Collaborative_Filtering)
    - [Recommendations with Neighborhood Based Collaborative Filtering in Code](#Recom_with_Collab_Filter_in_Code)
  - [Content-based Recommendations](#Content_based_Recommendations)
  - [Types of Ratings](#Types_of_Ratings) 
  
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


| ***Knowledge Based*** Recommendations     | ***Collaborative Filtering Based*** Recommendations    | ***Content Based*** Recommendations |
| :------------- | :------------- | :------------- |
| Analysis based on explicit knowledge about the item assortment, user preferences, and recommendation criteria    | Analysis based on connections between users and items   | Analysis based on the content of each item and finds similar items 
| No cold start (ramp-up) problems, however there are rules for knowledge aquisition     | uses the collaboration of user-item interactions       | Requires thorough knowledge of each item in order to find similar items 
|The idea here is to recommend items based on knowlege about items which meet user specifications | The idea here is to recommend items on similar user interests. Use ratings from many users across items in a collaborative way. | The idea here is to recommend similar items to the ones you liked before. Often the similarities are related to item descriptions or purpose.
| Example: Common for luxury or rare purchases (cars, homes, jewelery). Get back items which fullfill certain criteria (e.g., "the maximum price of the car is X")       | Example: "I liked the new Star Wars Film. You should go to the cinema."       | Example: You like Start Wars --> Recommend Avatar  
|In knowledge based recommendations, users provide information about the types of recommendations they would like back. |Similarity Measurement via correlation coefficients, euclidian distance | Similarity Measurement via correlation coefficients, euclidian distance, cosine similarity (similarity matrix), TF-IDF (e.g. in case of filtering out the genre from text)

***Business Cases For Recommendations***: 4 ideas to successfully implement recommendations to drive revenue, which include:

- Relevance
- Novelty
- Serendipity
- Increased Diversity

# Movie Tweeting Datasets <a name="Movie_Tweeting"></a>

Some Links:
 - [A Github account set up for MovieTweetings](https://github.com/sidooms/MovieTweetings)
 - [A slide deck by Simon Doom about MovieTweetings](https://www.slideshare.net/simondooms/rec-syschallenge2014intro?next_slideshow=1)

- Open notebook ```./notebooks/Introduction to the Recommendation Data.ipynb``` to check the ETL pipeline

# Knowledge based Recommendations <a name="Knowledge_based_recommendations"></a>

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
        INPUTS:
        ------------
            movies - the movies dataframe
            reviews - the reviews dataframe

        OUTPUTS:
        ------------
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
        INPUTS:
        ------------
            user_id - the user_id (str) of the individual you are making recommendations for
            n_top - an integer of the number recommendations you want back
            ranked_movies - a pandas dataframe of the already ranked movies based on avg rating, count, and time

        OUTPUTS:
        ------------
            top_movies - a list of the n_top recommended movies by movie title in order best to worst
        '''

        top_movies = list(ranked_movies['movie'][:n_top])

        return top_movies


  def popular_recs_filtered(user_id, n_top, ranked_movies, years=None, genres=None):
        '''
        INPUTS:
        ------------
            user_id - the user_id (str) of the individual you are making recommendations for
            n_top - an integer of the number recommendations you want back
            ranked_movies - a pandas dataframe of the already ranked movies based on avg rating, count, and time
            years - a list of strings with years of movies
            genres - a list of strings with genres of movies

        OUTPUTS:
        ------------
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

# Collaborative Filtering based Recommendation <a name="Collaborative_Filtering_Based"></a>
- A method of making recommendations based on ***using the collaboration of user-item interactions***

    ![image3]
- We need only: ***Information about how users and items interact with one another***
- Examples of Data Collaborative Filtering:
    - Item ratings for each user
    - Item liked by user or not
    - Item used by user or not
- Within ***Collaborative Filtering***, there are two main branches:

    - ***Model Based*** Collaborative Filtering
    - ***Neighborhood Based*** Collaborative Filtering

## Neighborhood Based Collaborative Filtering <a name="Neighborhood_Based_Collaborative_Filtering"></a>
- We are interested in finding individuals that are closely related to one another
- In doing so we need ***Similarity Metrics***
- Similarity between two users (or two items) can be analyzed via:

    - Pearson's correlation coefficient
    - Spearman's correlation coefficient
    - Kendall's Tau
    - Euclidean Distance
    - Manhattan Distance

    

    ![image1]

- Open notebook ```./notebooks/Measuring_Simularity.ipynb```

- ***Pearson's Correlation***
    - Statistical relationship between two continuous variables.  
    - It is known as the best method of measuring the association between variables
    - It is based on the method of covariance.  
    - It gives information about the magnitude of correlation, as well as the direction of the relationship

    - <img src="https://render.githubusercontent.com/render/math?math=CORR(x, y) = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i-\bar{x})^2}\sqrt{\sum_{i=1}^{n}(y_i-\bar{y})^2}} " height="80px">

    where 

    - <img src="https://render.githubusercontent.com/render/math?math=\bar{x} = \frac{1}{n}\sum_{i=1}^{n}x_i" height="40px">

    ```
    def pearson_corr(x, y):
        '''
        INPUTS:
        ------------
            x - an array of matching length to array y
            y - an array of matching length to array x

        OUTPUTS:
        ------------
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
        INPUTS:
        ------------
            x - an array of matching length to array y
            y - an array of matching length to array x

        OUTPUTS:
        ------------
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
        INPUTS:
        ------------
            x - an array of matching length to array y
            y - an array of matching length to array x

        OUTPUTS:
        ------------
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
        INPUTS:
        ------------
            x - an array of matching length to array y
            y - an array of matching length to array x

        OUTPUTS:
        ------------
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
        INPUTS:
            x - an array of matching length to array y
            y - an array of matching length to array x

        OUTPUTS:
        ------------
            manhat - the manhattan distance between x and y
        '''  
        
        return sum(abs(e - s) for s, e in zip(x, y))
    ```

## Recommendations with Neighborhood Based Collaborative Filtering in Code <a name="Recom_with_Collab_Filter_in_Code"></a>

- Open notebook ```./notebooks/Collaborative Filtering.ipynb```
- ***User-User Based Collaborative Filtering***

    ```
    import progressbar
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
        INPUTS:
        ------------
            user_id - the user_id of an individual as int

        OUTPUTS:
        ------------
            movies - an array of movies the user has watched
        '''
            movies = user_by_movie.loc[user_id][user_by_movie.loc[user_id].isnull() == False].index.values

        return movies

    def create_user_movie_dict():
        '''
        INPUTS: 
        ------------
            None

        OUTPUTS: 
        ------------
            movies_seen - a dictionary where each key is a user_id and the value is an array of movie_ids
        '''

        n_users = user_by_movie.shape[0]
        movies_seen = dict()

        cnter = 0
        bar = progressbar.Progressbar(maxval=n_user+1), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()

        for user1 in range(1, n_users+1):
            
            # assign list of movies to each user key
            movies_seen[user1] = movies_watched(user1)

            cnter+=1
            bar.update(cnter)
        
        bar.finish()
        
        return movies_seen
        
    movies_seen = create_user_movie_dict()
    ```

    ```
    # Remove individuals who have watched 2 or fewer movies - don't have enough data to make recs
    def create_movies_to_analyze(movies_seen, lower_bound=2):
        '''
        INPUTS:
        ------------  
            movies_seen - a dictionary where each key is a user_id and the value is an array of movie_ids
            lower_bound - (an int) a user must have more movies seen than the lower bound to be added to the movies_to_analyze dictionary

        OUTPUTS:
        ------------ 
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
        INPUTS:
        ------------
            user1 - int user_id
            user2 - int user_id

        OUTPUTS:
        ------------
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
    In the denominator of the correlation coefficient, we calculate the standard deviation for each user's ratings.  For example: The ratings for user 2 are all the same rating on the movies that match with user 104.  Therefore, the standard deviation is 0.  Because a 0 is in the denominator of the correlation coefficient, we end up with a **NaN** correlation coefficient.  Therefore, a different approach (e.g. euclidian distance approach) is likely better for this particular situation.
    ```
    def compute_euclidean_dist(user1, user2):
        '''
        INPUTS:
        ------------
            user1 - int user_id
            user2 - int user_id

        OUTPUTS:
        ------------
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
    df_dists = pd.read_pickle("dists.p")
    ```
- ***Using the Nearest Neighbors to Make Recommendations***

    - In the previous question, you read in df_dists. Therefore, you have a measure of distance between each user and every other user. This dataframe holds every possible pairing of users, as well as the corresponding euclidean distance.

    - Because of the NaN values that exist within the correlations of the matching ratings for many pairs of users, as we discussed above, we will proceed using df_dists. You will want to find the users that are 'nearest' each user. Then you will want to find the movies the closest neighbors have liked to recommend to each user.

    ***A certain bunch of objects***:
    - df_dists (to obtain the neighbors)
    - user_items (to obtain the movies the neighbors and users have rated)
    - movies (to obtain the names of the movies)

    The following functions below allow you to find the ***recommendations for any user***. There are five functions which you will need:

    - find_closest_neighbors - this returns a list of user_ids from closest neighbor to farthest neighbor using euclidean distance

    - movies_liked - returns an array of movie_ids

    - movie_names - takes the output of movies_liked and returns a list of movie names associated with the movie_ids

    - make_recommendations - takes a user id and goes through closest neighbors to return a list of movie names as recommendations

    - all_recommendations = loops through every user and returns a dictionary of with the key as a user_id and the value as a list of movie recommendations


    ```
    def find_closest_neighbors(user):
        '''
        INPUTS:
        ------------
            user - (int) the user_id of the individual you want to find the closest users
        
        OUTPUTS:
        ------------
            closest_neighbors - an array of the id's of the users sorted from closest to farthest away
        '''
        # I treated ties as arbitrary and just kept whichever was easiest to keep using the head method
        # You might choose to do something less hand wavy
        
        closest_users = df_dists[df_dists['user1']==user].sort_values(by='eucl_dist').iloc[1:]['user2']
        closest_neighbors = np.array(closest_users)
        
        return closest_neighbors
        
        
    def find_closest_neighbors(user):
        '''
        INPUTS:
        ------------
            user - (int) the user_id of the individual you want to find the closest users
        
        OUTPUTS:
        ------------
            closest_neighbors - an array of the id's of the users sorted from closest to farthest away
        '''
        # I treated ties as arbitrary and just kept whichever was easiest to keep using the head method
        # You might choose to do something less hand wavy
        
        closest_users = df_dists[df_dists['user1']==user].sort_values(by='eucl_dist').iloc[1:]['user2']
        closest_neighbors = np.array(closest_users)
        
        return closest_neighbors
        
        
    def movies_liked(user_id, min_rating=7):
        '''
        INPUTS:
        ------------
            user_id - the user_id of an individual as int
            min_rating - the minimum rating considered while still a movie is still a "like" and not a "dislike"

        OUTPUTS:
        ------------
            movies_liked - an array of movies the user has watched and liked
        '''

        movies_liked = np.array(user_items.query('user_id == @user_id and rating > (@min_rating -1)')['movie_id'])
        
        return movies_liked


    def movie_names(movie_ids):
        '''
        INPUTS:
        ------------
            movie_ids - a list of movie_ids
        
        OUTPUTS:
        ------------
            movies - a list of movie names associated with the movie_ids
        '''

        movie_lst = list(movies[movies['movie_id'].isin(movie_ids)]['movie'])
    
        return movie_lst
        

    def make_recommendations(user, num_recs=10):
        ''' - Make recommendations for user
            - movies_seen - movies seen by user
            - closest_neighbors - the closest neighbours of user
            - neighbs_likes - checkout the movies which were liked by a neighbour
            - np.setdiff1d - a nice way to compare two numpy arrays and 
                           - get the ones of the array that aren`t in the other 
                           - (it gives back array elements of neighbs_likes which are not in movies_seen)
        INPUTS:
        ------------
            user - (int) a user_id of the individual you want to make recommendations for
            num_recs - (int) number of movies to return

        OUTPUTS:
        ------------
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
        ''' Make recommendations for all users
            Pull only unique users from first column --> users = np.unique(df_dists['user1']) 

        INPUTS:
        ------------ 
            num_recs (int) the (max) number of recommendations for each user

        OUTPUTS:
        ------------
            all_recs - a dictionary where each key is a user_id and the value is an array of recommended movie titles
        '''
        
        # All the users we need to make recommendations for
        users = np.unique(df_dists['user1'])
        n_users = len(users)
        
        #Store all recommendations in this dictionary
        all_recs = dict()

        cnter = 0
        bar = progressbar.Progressbar(maxval=n_user+1, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        
        # Make the recommendations for each user
        for user in users:
            all_recs[user] = make_recommendations(user, num_recs)
            
            cnter+=1
            bar.update(cnter)
        
    bar.finish()
    
    return all_recs

    all_recs = all_recommendations(10)
    ```

# Content-based Recommendations <a name="Content_based_Recommendations"></a>
- The idea here is to recommend similar items to the ones you liked before. The system first finds the similarity between all pairs of articles and then uses the articles most similar to the articles already evaluated by a user to generate a list of recommendations.
- When to use? For example when there are only a few users. For example with collaborative filtering it could be impossible to find similar users. 
- Remember: In collaborative filtering, you are using the ***connections of users and items***. In content based techniques, you are using ***information about the users and items, but not connections***.

    ![image4]

- Open notebook ```./notebooks/Content_Based_Recommendation.ipynb```

    ```
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import defaultdict
    from IPython.display import HTML
    import progressbar
    import tests as t
    import pickle

    %matplotlib inline

    # Read in the datasets
    movies = pd.read_csv('movies_clean.csv')
    reviews = pd.read_csv('reviews_clean.csv')

    del movies['Unnamed: 0']
    del reviews['Unnamed: 0']

    all_recs = pickle.load(open("all_recs.p", "rb"))
    ```
    Even in the content based recommendation, you will often use collaborative filtering. You are finding items that are similar and making recommendations of new items based on the ***highest ratings of a user***. Because you are still using the user ratings of an item, this is an example of a blend between content and collaborative filtering based techniques.
    ```
    # create a dataframe similar to reviews, but ranked by rating for each user
    ranked_reviews = reviews.sort_values(by=['user_id', 'rating'], ascending=False)
    ranked_reviews.head(20)
    ```
    ***Similyrity matrix***: We can perform the dot product on a matrix of movies with content characteristics to provide a movie by movie matrix where each cell is ***an indication of how similar two movies are to one another***. 
    It's a matrix of similarities between items (movies) based only on the content related to those movies (year and genre). The similarity matrix is completely created using only the items (movies). There is no information used about the users implemented. For any movie, you would be able to determine the most related additional movies based only on the genre and the year of the movie. This is the premise of how a completely content based recommendation would be made.
    
    In the below image, you can see that movies 1 and 8 are most similar, movies 2 and 8 are most similar and movies 3 and 9 are most similar for this subset of the data. The diagonal elements of the matrix will contain the similarity of a movie with itself, which will be the largest possible similarity (which will also be the number of 1's in the movie row within the orginal movie content matrix.

    ![image5]

    ```
    # Subset of movies 
    # movie_content is only using the dummy variables for each genre and the 3 century based year dummy columns
    movie_content = np.array(movies.iloc[:, 4:])

    # Take the dot product to obtain a movie x movie matrix of similarities
    dot_prod_movies = movie_content.dot(np.transpose(movie_content))
    ```

    ```
    def find_similar_movies(movie_id):
        '''
        INPUTS:
        ------------
            movie_id - a movie_id 

        OUTPUTS:
        ------------
            similar_movies - an array of the most similar movies by title
        '''

        # find the row of each movie id
        movie_idx = np.where(movies['movie_id'] == movie_id)[0][0]
        
        # find the most similar movie indices - to start I said they need to be the same for all content
        similar_idxs = np.where(dot_prod_movies[movie_idx] == np.max(dot_prod_movies[movie_idx]))[0]
        
        # pull the movie titles based on the indices
        similar_movies = np.array(movies.iloc[similar_idxs, ]['movie'])
        
        return similar_movies
        
        
    def get_movie_names(movie_ids):
        '''
        INPUTS:
        ------------
            movie_ids - a list of movie_ids

        OUTPUTS:
        ------------
            movies - a list of movie names associated with the movie_ids
        '''

        movie_lst = list(movies[movies['movie_id'].isin(movie_ids)]['movie'])
    
        return movie_lst


    def make_recs():
        '''
        INPUT
        None
        OUTPUT
        recs - a dictionary with keys of the user and values of the recommendations
        '''
        # Create dictionary to return with users and ratings
        recs = defaultdict(set)
        # How many users for progress bar
        n_users = len(users_who_need_recs)

        
        # Create the progressbar
        cnter = 0
        bar = progressbar.ProgressBar(maxval=n_users+1, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        
        # For each user
        for user in users_who_need_recs:
            
            # Update the progress bar
            cnter+=1 
            bar.update(cnter)

            # Pull only the reviews the user has seen
            reviews_temp = ranked_reviews[ranked_reviews['user_id'] == user]
            
            # Look at each of the movies (highest ranked first), 
            movies_temp = np.array(reviews_temp['movie_id'])
            
            # pull the movies the user hasn't seen that are most similar
            movie_names = np.array(get_movie_names(movies_temp))
            
            # These will be the recommendations - continue until 10 recs 
            # or you have depleted the movie list for the user
            for movie in movies_temp:
                rec_movies = find_similar_movies(movie)
                temp_recs = np.setdiff1d(rec_movies, movie_names)
                recs[user].update(temp_recs)

                # If there are more than 
                if len(recs[user]) > 9:
                    break

        bar.finish()
        
        return recs
    ```
    ```
    # Explore recommendations
    users_without_all_recs = []
    users_with_all_recs = []
    no_recs = []
    for user, movie_recs in recs.items():
        if len(movie_recs) < 10:
            users_without_all_recs.append(user)
        if len(movie_recs) > 9:
            users_with_all_recs.append(user)
        if len(movie_recs) == 0:
            no_recs.append(user)
    ```
    ```
    # Closer look at individual user characteristics
    user_items = reviews[['user_id', 'movie_id', 'rating']]
    user_by_movie = user_items.groupby(['user_id', 'movie_id'])['rating'].max().unstack()

    def movies_watched(user_id):
        '''
        INPUT:
        user_id - the user_id of an individual as int
        OUTPUT:
        movies - an array of movies the user has watched
        '''
        movies = user_by_movie.loc[user_id][user_by_movie.loc[user_id].isnull() == False].index.values

        return movies

    movies_watched(189)
    ```

    ```
    cnter = 0
    print("Some of the movie lists for users without any recommendations include:")
    for user_id in no_recs:
        print(user_id)
        print(get_movie_names(movies_watched(user_id)))
        cnter+=1
        if cnter > 10:
            break
    ```
# Types of Ratings: <a name="Types_of_Ratings"></a>
- If you are in control of choosing your rating scale, think of what might be most beneficial to your scenario. If you are working alongside a team TO design the interfaces for how data will be collected, there are number of ideas to keep in mind. 
- Overview of [ratings](https://cxl.com/blog/survey-response-scales/)

    ![image8]
    
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
Recommendation Engines
* [Essentials of recommendation engines: content-based and collaborative filtering](https://towardsdatascience.com/essentials-of-recommendation-engines-content-based-and-collaborative-filtering-31521c964922)
* [AirBnB uses Embeddings for Recommendations](https://medium.com/airbnb-engineering/listing-embeddings-for-similar-listing-recommendations-and-real-time-personalization-in-search-601172f7603e)
* [Location-Based Recommendation Systems](https://link.springer.com/referenceworkentry/10.1007%2F978-3-319-17885-1_1580)

    ![image6]
* [Deep learning for recommender systems](https://ebaytech.berlin/deep-learning-for-recommender-systems-48c786a20e1a)

    ![image7]

Git/Github
* [GitFlow](https://datasift.github.io/gitflow/IntroducingGitFlow.html)
* [A successful Git branching model](https://nvie.com/posts/a-successful-git-branching-model/)
* [5 types of Git workflows](https://buddy.works/blog/5-types-of-git-workflows)

Docstrings, DRY, PEP8
* [Python Docstrings](https://www.geeksforgeeks.org/python-docstrings/)
* [DRY](https://www.youtube.com/watch?v=IGH4-ZhfVDk)
* [PEP 8 -- Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/)
