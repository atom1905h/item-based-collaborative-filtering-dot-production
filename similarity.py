from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import numpy as np
from joblib import Parallel, delayed
import multiprocessing

def to_dataframe(similarity_matrix, matrix):
    similarity_df = pd.DataFrame(similarity_matrix,
                                index=matrix.index, columns=matrix.index)
    return similarity_df

def cosine_sim(matrix):
    cosine_similarity_matrix = cosine_similarity(matrix)

    return cosine_similarity_matrix

def euclidean_sim(matrix):
    euclidean_similarity_matrix = 1/(1+euclidean_distances(matrix))
    
    return euclidean_similarity_matrix

def manhattan_sim(matrix):
    manhattan_similarity_matrix = 1/(1+manhattan_distances(matrix))
    
    return manhattan_similarity_matrix

def jaccard_sim(matrix):
    jaccard_matrix = 1-pdist(matrix, 'jaccard')
    jaccard_similarity_matrix = squareform(jaccard_matrix)
    np.fill_diagonal(jaccard_similarity_matrix, 1)

    return jaccard_similarity_matrix

def pearson_sim(matrix):
    pearson_similarity_matrix = np.corrcoef(matrix)

    return pearson_similarity_matrix

def msd_similarity(movie1, movie2):
    common_users = np.logical_and(~np.isnan(movie1), ~np.isnan(movie2))
    if np.sum(common_users) == 0:
        return 0

    msd = np.sum(np.square(movie1[common_users] - movie2[common_users])) / np.sum(common_users)
    similarity = 1 / (1 + msd)
    return similarity

def compute_similarity(i, j, df):
    movie1 = df.iloc[i].values
    movie2 = df.iloc[j].values
    similarity = msd_similarity(movie1, movie2)
    return i, j, similarity

def msd_similarity_matrix_parallel(df):
    num_movies = len(df)
    similarity_matrix = np.zeros((num_movies, num_movies))
    
    num_cores = multiprocessing.cpu_count()

    results = Parallel(n_jobs=num_cores)(
        delayed(compute_similarity)(i, j, df) for i in range(num_movies) for j in range(i, num_movies)
    )

    for i, j, similarity in results:
        similarity_matrix[i, j] = similarity
        similarity_matrix[j, i] = similarity

    return similarity_matrix

# 결측치 없어야 함
def proximity(x, y):
    exp_value = np.exp(-np.abs(x - y))
    z = 1 - 1 / (1 + exp_value)
    return z

def significance(x, y, rm):
    exp_value = np.exp(-np.abs(x - rm) * np.abs(y - rm))
    z = 1 / (1 + exp_value)
    return z

def singularity(x, y, muj):
    exp_value = np.exp(-np.abs((x + y) / 2 - muj))
    z = 1 - 1 / (1 + exp_value)
    return z

def median_vector(M, take='column'):
    M = np.array(M)
    n = M.shape[0]
    mv = np.zeros(n) if take == 'column' else np.zeros((n, 1))
    md=np.nanmedian(M)
    for i in range(n) if take == 'column' else range(n):
            mv[i] = md
    return mv

def mean_vector(M, take='column'):
    M = np.array(M)
    n = M.shape[0]
    mv = np.zeros(n) if take == 'column' else np.zeros((n, 1))
    mn=np.nanmean(M)
    for i in range(n) if take == 'column' else range(n):
            mv[i] = mn
    return mv

def pss(M):
    M = np.array(M)
    n = M.shape[0]
    af_matrix = np.ones((n, n))

    for i in range(n - 1):
        for j in range(i + 1, n):
            if np.isnan(np.sum(M[i, :])) or np.isnan(np.sum(M[j, :])):
                pss = 0
                af_matrix[i, j] = af_matrix[j, i] = pss
            else:
                ui = M[i, :]
                uj = M[j, :]
                rm = median_vector(ui, take='column')
                muj = mean_vector(uj, take='column')
                pss = np.nanmean(proximity(ui, uj) * significance(ui, uj, rm=rm) * singularity(ui, uj, muj=muj))
                af_matrix[i, j] = af_matrix[j, i] = pss

    return af_matrix