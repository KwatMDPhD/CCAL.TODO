def _update_w(ma, wm, hm):

    return wm * (ma @ hm.T) / (wm @ hm @ hm.T)
