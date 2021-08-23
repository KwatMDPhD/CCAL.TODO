def get_p_value(nu, ra_, di):

    if di == "<":

        bo_ = ra_ <= nu

    elif di == ">":

        bo_ = nu <= ra_

    return max(1, bo_.sum()) / ra_.size
