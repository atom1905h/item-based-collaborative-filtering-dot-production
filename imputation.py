from sklearn.impute import KNNImputer

def fill_zero(matrix):
    matrix = matrix.fillna(0)

    return matrix

def fill_mean_1(matrix):
    for col in matrix.index:
        value = matrix.loc[col].mean()
        matrix.loc[col] = matrix.loc[col].fillna(value)

    return matrix

def fill_mean_2(matrix):
    for col in matrix.columns:
        value = matrix[col].mean()
        matrix[col] = matrix[col].fillna(value)

    return matrix

def fill_median_1(matrix):
    for col in matrix.index:
        value = matrix.loc[col].median()
        matrix.loc[col] = matrix.loc[col].fillna(value)

    return matrix

def fill_median_2(matrix):
    for col in matrix.columns:
        value = matrix[col].median()
        matrix[col] = matrix[col].fillna(value)

    return matrix

def fill_mode_1(matrix):
    for col in matrix.index:
        value = matrix.loc[col].mode()[0]
        matrix.loc[col] = matrix.loc[col].fillna(value)

    return matrix

def fill_mode_2(matrix):
    for col in matrix.columns:
        value = matrix[col].mode()[0]
        matrix[col] = matrix[col].fillna(value)

    return matrix

def fill_knn(matrix, n):
    knn_imputer = KNNImputer(n_neighbors=n)
    user_item_matrix_imputed = knn_imputer.fit_transform(matrix)

    return user_item_matrix_imputed