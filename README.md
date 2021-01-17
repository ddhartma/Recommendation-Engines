[image1]: assets/ab_test.png "image1"

# Recommendation Engines
Let's create Recommendation Engines for Movie Tweetings

## Outline
- [Recommendation Engines](#Recommendation_Engines)
  - [What's Ahead](#Whats_Ahead)
  - [Movie Tweeting Datasets](#Movie_Tweeting)
  - [Knowledge based recommendations](#Knowledge_based_recommendations)


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

## Knowledge based recommendations <a name="Knowledge_based_recommendations"></a>

1. ***Rank Based Recommendations***:
    - Recommendation based on highest ratings, most purchases, most listened to, etc.
    - For example: Recommend most popular film of all film categories

2. ***Knowledge Based Recommnedations***:
    - Knowledge about item or user preferences are used to make a recommendation
    - For example: Filter first for a certain film category, then recommend most popular film out of this category

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
$ git clone https://github.com/ddhartma/Disaster-Response-Pipeline-Project.git
```

- Change Directory
```
$ cd Disaster-Response-Pipeline-Project
```

- Create a new Python environment, e.g. ds_ndp. Inside Git Bash (Terminal) write:
```
$ conda create --name ds_ndp
```

- Activate the installed environment via
```
$ conda activate ds_ndp
```

- Install the following packages (via pip or conda)
```
numpy = 1.17.4
pandas = 0.24.2
scikit-learn = 0.20
pipelinehelper = 0.7.8
```
Example via pip:
```
pip install numpy
pip install pandas
pip install scikit-learn==0.20
pip install pipelinehelper
```
scikit-learn==0.20 is needed for sklearns dictionary output (output_dict=True) for the classification_report. Earlier versions do not support this.


Link1 to [pipelinehelper](https://github.com/bmurauer/pipelinehelper)

Link2 to [pipelinehelper](https://stackoverflow.com/questions/23045318/scikit-grid-search-over-multiple-classifiers)

- Check the environment installation via
```
$ conda env list
```

### Switch the pipelines
- Active pipeline at the moment: pipeline_1 (Fast training/testing) pipeline
- More sophisticated pipelines start with pipeline_2.
- Model training has been done with ```pipeline_2```.
- In order to switch between pipelines or to add more pipelines open train.classifier.py. Adjust the pipelines in `def main()` which should be tested (only one or more are possible) via the list ```pipeline_names```.

```
def main():
   ...

    if len(sys.argv) == 3:
        ...

        # start pipelining, build the model
        pipeline_names = ['pipeline_1', 'pipeline_2']
```


### Run the web App

1. Run the following commands in the project's root directory to set up your database and model.

    To run ETL pipeline that cleans data and stores in database


        python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

    To run ML pipeline that trains classifier and saves

        python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl


2. Run the following command in the app's directory to run your web app

        python run.py




3. Go to http://0.0.0.0:3001/

## Acknowledgments <a name="Acknowledgments"></a>
* This project is part of the Udacity Nanodegree program 'Data Science'. Please check this [link](https://www.udacity.com) for more information.

## Further Links <a name="Further_Links"></a>
## Links
* [Correlation does not imply causation](https://en.wikipedia.org/wiki/Correlation_does_not_imply_causation)
* [17 Statistical Hypothesis Tests in Python ](https://machinelearningmastery.com/statistical-hypothesis-tests-in-python-cheat-sheet/)

Git/Github
* [GitFlow](https://datasift.github.io/gitflow/IntroducingGitFlow.html)
* [A successful Git branching model](https://nvie.com/posts/a-successful-git-branching-model/)
* [5 types of Git workflows](https://buddy.works/blog/5-types-of-git-workflows)

Docstrings, DRY, PEP8
* [Python Docstrings](https://www.geeksforgeeks.org/python-docstrings/)
* [DRY](https://www.youtube.com/watch?v=IGH4-ZhfVDk)
* [PEP 8 -- Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/)

Data Visualization
* [10 Python Data Visualization Libraries for Any Field | Mode](https://mode.com/blog/python-data-visualization-libraries/)
* [5 Quick and Easy Data Visualizations in Python with Code](https://towardsdatascience.com/5-quick-and-easy-data-visualizations-in-python-with-code-a2284bae952f)
* [The Best Python Data Visualization Libraries](https://www.fusioncharts.com/blog/best-python-data-visualization-libraries/)
