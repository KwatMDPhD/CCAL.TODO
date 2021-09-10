def _update_h(ma, wm, hm):

    return hm * (wm.T @ ma) / (wm.T @ wm @ hm)
