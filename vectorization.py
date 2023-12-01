import pandas as pd


def item_rating(df):
    movie_rating = df.pivot_table(values='rating', index='movieId', columns='userId')

    return movie_rating

def item_difference_release_rating(df):
    df['time_difference_seconds'] = (pd.to_datetime(df['timestamp'], unit='s', utc=True) - pd.to_datetime(df['year'], format='%Y', utc=True)).dt.total_seconds()
    movie_time = df.pivot_table(values='time_difference_seconds', index='movieId', columns='userId')

    return movie_time

def item_rating_average_deviation(df):
    user_avg_ratings = df.groupby('userId')['rating'].mean()
    df['rating_diff_user_avg'] = df['rating'] - df['userId'].map(user_avg_ratings)
    movie_rating_average_deviation = df.pivot_table(values='rating_diff_user_avg', index='movieId', columns='userId')
    
    return movie_rating_average_deviation

def item_rating_or_not(df):
    df['rating_or_not'] = df['rating'].apply(lambda x: 1 if x > 0 else 0)
    movie_rating_or_not = df.pivot_table(values='rating_or_not', index='movieId', columns='userId', fill_value=0)

    return movie_rating_or_not

def item_tag_or_not(df):
    df['tag_or_not'] = df['tags_timestamp'].apply(lambda x: 1 if x > 0 else 0)
    movie_tag_or_not = df.pivot_table(values='tag_or_not', index='movieId', columns='userId', fill_value=0)

    return movie_tag_or_not

def item_rating_tag_or_not(df):
    df['interaction'] = df.apply(lambda row: 4 if pd.notna(row['rating']) and pd.notna(row['tag'])
                             else 3 if pd.notna(row['rating'])
                             else 2 if pd.notna(row['tag'])
                             else 1, axis=1)
    movie_rating_tag_or_not = df.pivot_table(values='interaction', index='movieId', columns='userId', fill_value=1)

    return movie_rating_tag_or_not