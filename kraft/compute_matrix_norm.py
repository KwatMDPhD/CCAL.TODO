from numpy.linalg import norm


def compute_matrix_norm(M):

    return norm(M, ord="fro")
