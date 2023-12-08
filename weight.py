import pandas as pd
import numpy as np

def aggregate(ratings_df, mode): 
    if mode == 'movie':
        aggregated_df = pd.DataFrame({
            'num_users': ratings_df.groupby('movieId')['userId'].nunique(),
            'avg_rating': ratings_df.groupby('movieId')['rating'].mean(), 
            'std_rating': ratings_df.groupby('movieId')['rating'].std()
        }).reset_index()
    elif mode == 'user':
        aggregated_df = pd.DataFrame({
            'num_movies': ratings_df.groupby('userId')['movie_id'].nunique(),
            'avg_rating': ratings_df.groupby('userId')['rating'].mean(), 
            'std_rating': ratings_df.groupby('userId')['rating'].std()
        }).reset_index()

    return aggregated_df

def get_steam_rating(movie_statistics):
    movie_statistics['steam_rating'] = movie_statistics['avg_rating'] - (movie_statistics['avg_rating'] - 2.5)*np.power(2, -np.log10(movie_statistics['num_users']))
    return movie_statistics

def weight_df(df):
    movie_statistics = aggregate(df, 'movie')
    movie_statistics = get_steam_rating(movie_statistics)
    movie_statistics['normalized_user_count'] = (movie_statistics['num_users']/movie_statistics['num_users'].max())*5
    return movie_statistics
