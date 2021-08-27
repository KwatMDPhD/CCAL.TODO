from ..array import check_is_extreme

def get_extreme(se, **ke):

    ex_ = check_is_extreme(se.values, **ke)

    la_ = se.index.values

    for di in ["<", ">"]:

        la_[ex_, di]

    return ex_
