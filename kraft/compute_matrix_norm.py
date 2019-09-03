from numpy.linalg import norm


def compute_matrix_norm(matrix):

    return norm(matrix, ord="fro")
