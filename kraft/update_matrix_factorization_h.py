def update_mf_h(V, W, H):

    return H * (W.T @ V) / (W.T @ W @ H)
