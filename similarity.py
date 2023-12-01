from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import numpy as np

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