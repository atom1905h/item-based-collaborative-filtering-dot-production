import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN

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

def item_tags_count(df): # 결측치 있음
    df['combined'] = (df['tag'].fillna('') + df['genres'].fillna('')).astype(str)
    df['tag_count'] = df['combined'].apply(lambda x: len(x.split('|')))
    movie_tag_count = df.pivot_table(values='tag_count', index='movieId', columns='userId')

    return movie_tag_count

def item_tf_idf_sum(df): # 결측치 있음
    df['combined'] = (df['tag'].fillna('') + df['genres'].fillna('')).astype(str)
    vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split('|'))
    tfidf_matrix = vectorizer.fit_transform(df['combined'])
    df['tf-idf'] = tfidf_matrix.sum(axis=1)
    movie_tf_idf = df.pivot_table(values='tf-idf', index='movieId', columns='userId')

    return movie_tf_idf

def item_tf_idf(df): #(9737, 9737) 결측치 없음
    df['combined'] = (df['tag'].fillna('') + df['genres'].fillna('')).astype(str)
    df = df.groupby('movieId')['combined'].apply(lambda x: '|'.join(set(x))).reset_index(name='combined_m')
    vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split('|'))
    tfidf_matrix = vectorizer.fit_transform(df['combined_m'])

    return tfidf_matrix

def item_cluster_kmean(df):
    df['time_difference_seconds'] = (pd.to_datetime(df['timestamp'], unit='s', utc=True) - pd.to_datetime(df['year'], format='%Y', utc=True)).dt.total_seconds()
    user_avg_ratings = df.groupby('userId')['rating'].mean()
    df['rating_diff_user_avg'] = df['rating'] - df['userId'].map(user_avg_ratings)
    scaler = MinMaxScaler()
    df_X = df[['rating','time_difference_seconds','rating_diff_user_avg']]
    df_X = df_X.fillna(df_X.mean())
    data_scale = scaler.fit_transform(df_X)
    model = KMeans(n_clusters = 7, random_state = 10)
    df['cluster'] = model.fit_predict(data_scale)
    movie_user_cluster = df.pivot_table(values='cluster', index='movieId', columns='userId')

    return movie_user_cluster

def item_cluster_dbscan(df):
    df['time_difference_seconds'] = (pd.to_datetime(df['timestamp'], unit='s', utc=True) - pd.to_datetime(df['year'], format='%Y', utc=True)).dt.total_seconds()
    user_avg_ratings = df.groupby('userId')['rating'].mean()
    df['rating_diff_user_avg'] = df['rating'] - df['userId'].map(user_avg_ratings)
    df_X = df[['rating','time_difference_seconds','rating_diff_user_avg']]
    df_X = df_X.fillna(df_X.mean())
    model = DBSCAN(n_clusters = 7, random_state = 10)
    df['cluster'] = model.fit_predict(df_X)
    movie_user_cluster = df.pivot_table(values='cluster', index='movieId', columns='userId')

    return movie_user_cluster