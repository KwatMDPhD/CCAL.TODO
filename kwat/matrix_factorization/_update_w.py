def _update_w(v, w, h):

    return w * (v @ h.T) / (w @ h @ h.T)
