from argparse import ArgumentParser
from imputation import fill_zero, fill_mean_1, fill_mean_2, fill_knn, fill_median_1, fill_median_2, fill_mode_1, fill_mode_2
from similarity import to_dataframe, cosine_sim, euclidean_sim, jaccard_sim, pearson_sim, manhattan_sim
from vectorization import item_rating, item_difference_release_rating, item_rating_average_deviation, item_rating_or_not, item_rating_tag_or_not, item_tag_or_not
from predict import predict_user_ratings
import pandas as pd

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--vectorization',type=str, default='item_rating')
    parser.add_argument('--imputation',type=str,default='fill_zero')
    parser.add_argument('--similarity', type=str, default='cosine')
    parser.add_argument('--user_id', type=int, default=1)
    args = parser.parse_args()

    return args

def recommend(vectorization, imputation, similarity, user_id):
    ratings = pd.read_csv('data/ratings_v1.csv')
    movies = pd.read_csv('data/movies_v1.csv')
    tags = pd.read_csv('data/tags.csv')
    tags.columns = ['userId', 'movieId', 'tag', 'tags_timestamp']
    grouped_tags = tags.groupby(['userId', 'movieId']).agg({'tag': '|'.join, 'tags_timestamp': 'mean'}).reset_index()
    df = pd.merge(ratings, movies, on='movieId', how='left')
    df = pd.merge(df, grouped_tags, on=['userId','movieId'], how='outer')

    user_ratings= df.pivot_table(values='rating', index='movieId', columns='userId')

    if vectorization=='item_rating':
        vector = item_rating(df)
    elif vectorization=='item_difference_release_rating':
        vector = item_difference_release_rating(df)
    elif vectorization=='item_rating_average_deviation':
        vector = item_rating_average_deviation(df)
    elif vectorization=='item_rating_or_not':
        vector = item_rating_or_not(df)
    elif vectorization=='item_rating_tag_or_not':
        vector = item_rating_tag_or_not(df)
    elif vectorization=='item_tag_or_not':
        vector = item_tag_or_not(df)

    if imputation=='fill_zero':
        vector = fill_zero(vector)
    elif imputation=='fill_mean_1':
        vector = fill_mean_1(vector)
    elif imputation=='fill_mean_2':
        vector = fill_mean_2(vector)
    elif imputation=='fill_knn':
        vector = fill_knn(vector)
    elif imputation=='fill_median_1':
        vector = fill_median_1(vector)
    elif imputation=='fill_median_2':
        vector = fill_median_2(vector)
    elif imputation=='fill_mode_1':
        vector = fill_mode_1(vector)
    elif imputation=='fill_mode_2':
        vector = fill_mode_2(vector)

    if similarity=='cosine':
         matrix = cosine_sim(vector)
    if similarity=='euclidean':
         matrix = euclidean_sim(vector)
    if similarity=='jaccard':
         matrix = jaccard_sim(vector)
    if similarity=='pearson':
         matrix = pearson_sim(vector)
    if similarity=='manhattan':
         matrix = manhattan_sim(vector)
    
    similarity_matrix = to_dataframe(matrix, vector)
    predicted_ratings = predict_user_ratings(user_id, user_ratings=user_ratings, item_similarity=similarity_matrix)
    predicted_ratings_sorted = predicted_ratings.sort_values(by='predicted_rating', ascending=False)
    top_10=predicted_ratings_sorted.head(10)['movieId'].tolist()
    top_10_movies = movies[movies['movieId'].isin(top_10)]

    print(top_10_movies['title'])

def main(args):
    recommend(**args.__dict__)
if __name__ == '__main__':
    
    args = parse_args()
    main(args)