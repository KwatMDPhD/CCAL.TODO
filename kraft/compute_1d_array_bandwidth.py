from statsmodels.nonparametric.kernel_density import KDEMultivariate


def compute_1d_array_bandwidth(_1d_array):

    return KDEMultivariate(_1d_array, "c").bw[0]
