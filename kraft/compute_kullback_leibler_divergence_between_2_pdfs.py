from numpy import log


def compute_kullback_leibler_divergence(pdf_1, pdf_0):

    return pdf_1 * log(pdf_1 / pdf_0)
