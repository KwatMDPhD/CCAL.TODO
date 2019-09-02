from statsmodels.nonparametric.kernel_density import KDEMultivariate


def compute_vector_bandwidth(_vector):

    return KDEMultivariate(_vector, "c").bw[0]
