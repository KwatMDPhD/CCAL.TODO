def _update_h(v, w, h):

    return h * (w.T @ v) / (w.T @ w @ h)
