from statsmodels.sandbox.stats.multicomp import multipletests


def get_q_value(pv_):

    return multipletests(pv_)[1]
