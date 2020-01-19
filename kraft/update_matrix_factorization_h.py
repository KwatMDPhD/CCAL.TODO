def update_matrix_factorization_h(V, W, H):

    return H * (W.T @ V) / (W.T @ W @ H)
