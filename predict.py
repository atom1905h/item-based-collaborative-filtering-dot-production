import numpy as np
import pandas as pd

def predict_user_unrated_ratings(user_id, user_ratings, item_similarity):
    pred_ratings=[]
    unrated_movies = user_ratings[user_ratings[user_id].isnull()][user_id].index.tolist()
    rated_movies =user_ratings[user_ratings[user_id].notnull()][user_id].index.tolist()
    user_ratings = user_ratings[user_id].loc[rated_movies].tolist()
    watched_user_y = np.array(user_ratings).reshape(-1, 1)
    for i in unrated_movies:
        pred_rating = np.matmul(item_similarity[i][rated_movies],watched_user_y)/(sum(item_similarity[i][rated_movies])+1)
        pred_ratings.append(pred_rating[0])
    result = pd.DataFrame({'movieId': unrated_movies, 'userId':user_id, 'predicted_rating': pred_ratings})

    return result

def predict_user_rated_ratings(user_id,user_ratings, item_similarity):
    pred_ratings = []
    rated_movies =user_ratings[user_ratings[user_id].notnull()][user_id].index.tolist()
    
    for i in rated_movies:
        rated_movies_except_i = [movie for movie in rated_movies if movie != i]
        user_rating = user_ratings[user_id].loc[rated_movies_except_i].tolist()
        watched_user_y = np.array(user_rating).reshape(-1, 1)
        pred_rating = np.matmul(item_similarity[i][rated_movies_except_i], watched_user_y)/(sum(item_similarity[i][rated_movies_except_i]) + 1)
        pred_ratings.append(pred_rating[0])
    result = pd.DataFrame({'movieId': rated_movies, 'userId':user_id, 'predicted_rating': pred_ratings})

    return result