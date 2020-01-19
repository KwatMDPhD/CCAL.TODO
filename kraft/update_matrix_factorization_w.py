def update_matrix_factorization_w(V, W, H):

    return W * (V @ H.T) / (W @ H @ H.T)
