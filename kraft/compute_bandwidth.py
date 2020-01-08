from statsmodels.nonparametric.kernel_density import KDEMultivariate


def compute_bandwidth(vector):

    return KDEMultivariate(vector, "c").bw[0]
