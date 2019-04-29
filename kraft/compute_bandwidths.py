from statsmodels.nonparametric.kernel_density import KDEMultivariate


def compute_bandwidths(variables):

    return KDEMultivariate(variables, "c" * len(variables)).bw
