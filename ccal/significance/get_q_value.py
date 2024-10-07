from statsmodels.sandbox.stats.multicomp import multipletests


def get_q_value(pv_):
    return multipletests(pv_, method="fdr_bh")[1]
