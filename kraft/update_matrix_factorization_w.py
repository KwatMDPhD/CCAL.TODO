def update_mf_w(V, W, H):

    return W * (V @ H.T) / (W @ H @ H.T)
