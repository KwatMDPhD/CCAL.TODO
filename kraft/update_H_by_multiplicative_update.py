def update_h_by_multiplicative_update(V, W, H):

    return H * (W.T @ V) / (W.T @ W @ H)
