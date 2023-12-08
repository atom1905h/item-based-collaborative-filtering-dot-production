from argparse import ArgumentParser
from imputation import fill_zero, fill_mean_1, fill_mean_2, fill_knn, fill_median_1, fill_median_2, fill_mode_1, fill_mode_2
from similarity import to_dataframe, cosine_sim, euclidean_sim, jaccard_sim, pearson_sim, manhattan_sim, msd_similarity_matrix_parallel, pss
from vectorization import item_rating, item_difference_release_rating, item_rating_average_deviation, item_rating_or_not, item_rating_tag_or_not, item_tag_or_not, item_tags_count, item_tf_idf, item_tf_idf_sum, item_cluster_kmean, item_cluster_dbscan, item_tree_combine_feature
from predict import predict_user_unrated_ratings, predict_user_rated_ratings
from weight import weight_df
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--vectorization',type=str, default='item_rating')
    parser.add_argument('--imputation',type=str,default='fill_zero')
    parser.add_argument('--similarity', type=str, default='cosine')
    parser.add_argument('--user_id', type=int, default=1)
    parser.add_argument('--weight', type=str, default='False')
    parser.add_argument('--weight_sd', type=str, default='num_user')
    args = parser.parse_args()

    return args

def recommend(vectorization, imputation, similarity, weight, weight_sd, user_id):
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
    elif vectorization=='item_tags_count':
        vector = item_tags_count(df)
    elif vectorization=='item_tf_idf_sum':
        vector = item_tf_idf_sum(df)
    elif vectorization=='item_tf_idf':
        vector = item_tf_idf(df)
    elif vectorization=='item_cluster_kmean':
        vector = item_cluster_kmean(df)
    elif vectorization=='item_cluster_dbscan':
        vector = item_cluster_dbscan(df)
    elif vectorization=='item_tree':
        vector = item_tree_combine_feature(df)

    if imputation=='fill_zero':
        vector1 = fill_zero(vector)
    elif imputation=='fill_mean_1':
        vector1 = fill_mean_1(vector)
    elif imputation=='fill_mean_2':
        vector1 = fill_mean_2(vector)
    elif imputation=='fill_knn':
        vector1 = fill_knn(vector,3)
    elif imputation=='fill_median_1':
        vector1 = fill_median_1(vector)
    elif imputation=='fill_median_2':
        vector1 = fill_median_2(vector)
    elif imputation=='fill_mode_1':
        vector1 = fill_mode_1(vector)
    elif imputation=='fill_mode_2':
        vector1 = fill_mode_2(vector)
    elif imputation=='no':
        vector1 = vector
        
    if similarity=='cosine':
         matrix = cosine_sim(vector1)
    elif similarity=='euclidean':
         matrix = euclidean_sim(vector1)
    elif similarity=='jaccard':
         matrix = jaccard_sim(vector1)
    elif similarity=='pearson':
         matrix = pearson_sim(vector1)
    elif similarity=='manhattan':
         matrix = manhattan_sim(vector1)
    elif similarity=='msd':
         matrix = msd_similarity_matrix_parallel(vector1)
    elif similarity=='pss':
         matrix = pss(vector1)
    elif similarity=='jmsd':
        matrix_j = jaccard_sim(vector1)
        matrix_m = msd_similarity_matrix_parallel(vector1)
        matrix = matrix_j*matrix_m
    elif similarity=='jpss':
        matrix_j = jaccard_sim(vector1)
        matrix_p = pss(vector1)
        matrix = matrix_j*matrix_p

    if vectorization=='item_tf_idf':
        similarity_matrix = pd.DataFrame(matrix,
                                index=movies['movieId'].unique().tolist(), columns=movies['movieId'].unique().tolist())
    else:
        similarity_matrix = to_dataframe(matrix, vector)

    user_list = df['userId'].unique().tolist()
    result = predict_user_rated_ratings(user_list[0], user_ratings=user_ratings, item_similarity=similarity_matrix)
    for i in user_list[1:]:
        predicted_ratings = predict_user_rated_ratings(i, user_ratings=user_ratings, item_similarity=similarity_matrix)
        if weight=='True':
            w = weight_df(df)
            predicted_ratings = pd.merge(predicted_ratings, w, on='movieId', how='left')
            if weight_sd=='num_user':
                predicted_ratings['predicted_rating'] = predicted_ratings['predicted_rating']*0.8 + predicted_ratings['normalized_user_count']*0.2
            elif weight_sd=='steam_rating':
                predicted_ratings['predicted_rating'] = predicted_ratings['predicted_rating']*0.8 + predicted_ratings['steam_rating']*0.2
        result= pd.concat([result, predicted_ratings],axis=0)

    result2=pd.merge(df, result, on=['userId','movieId'],how='left')
    result2_not_null = result2.dropna(subset=['rating'])
    true_y = np.array(result2_not_null['rating'])
    pred_y = np.array(result2_not_null['predicted_rating'])
    mae = mean_absolute_error(y_true=true_y, y_pred=pred_y)
    mse = mean_squared_error(y_true=true_y, y_pred=pred_y)
    rmse = np.sqrt(mse)
    
    if weight=='True':
        name = vectorization+"_"+imputation+"_"+ similarity+"_"+weight_sd
    else:
        name = vectorization+"_"+imputation+"_"+ similarity

    result_metric = pd.read_csv('result/result.csv')
    new_result_metric = pd.DataFrame(data=[[name, mae, mse, rmse]], columns=['name','mae','mse','rmse'])
    result_metric = result_metric.append(new_result_metric)
    result_metric.to_csv('result/result.csv', index=False)

    recommended_movies_df = pd.DataFrame(columns=['userId', 'movieId', 'title','genres','year'])
    for j in user_list:
        predicted_unrated_ratings = predict_user_unrated_ratings(j, user_ratings=user_ratings, item_similarity=similarity_matrix)
        if weight=='True':
            w = weight_df(df)
            predicted_unrated_ratings = pd.merge(predicted_unrated_ratings, w, on='movieId', how='left')
            if weight_sd=='num_user':
                predicted_unrated_ratings['predicted_rating'] = predicted_unrated_ratings['predicted_rating']*0.8 + predicted_unrated_ratings['normalized_user_count']*0.2
            elif weight_sd=='steam_rating':
                predicted_unrated_ratings['predicted_rating'] = predicted_unrated_ratings['predicted_rating']*0.8 + predicted_unrated_ratings['steam_rating']*0.2
        predicted_ratings_sorted = predicted_unrated_ratings.sort_values(by='predicted_rating', ascending=False)
        top_5=predicted_ratings_sorted.head(5)['movieId'].tolist()
        top_5_movies = movies[movies['movieId'].isin(top_5)][['movieId', 'title','genres','year']]
        top_5_movies['userId'] = j
        recommended_movies_df = recommended_movies_df.append(top_5_movies)
    
    recommended_movies_df.to_csv(f'result/{name}.csv',index=False)

def main(args):
    recommend(**args.__dict__)
if __name__ == '__main__':
    
    args = parse_args()
    main(args)