from statsmodels.nonparametric.kernel_density import KDEMultivariate


def get_bandwidth(nu_):

    return KDEMultivariate(nu_, "c").bw[0]
