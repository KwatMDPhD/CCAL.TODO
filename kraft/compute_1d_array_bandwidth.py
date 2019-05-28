from statsmodels.nonparametric.kernel_density import KDEMultivariate


def compute_1d_array_bandwidth(_1d_array, kdemultivariate_bw="normal_reference"):

    return KDEMultivariate((_1d_array,), "c", bw=kdemultivariate_bw).bw[0]
