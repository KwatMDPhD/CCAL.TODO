def get_p_value(nu, ra_, di):
    if di == "<":
        si_ = ra_ <= nu

    elif di == ">":
        si_ = nu <= ra_

    return max(1, si_.sum()) / ra_.size
