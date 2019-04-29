from numpy import log


def compute_kullback_leibler_divergence_between_2_pdfs(pdf_0, pdf_1):

    return pdf_0 * log(pdf_0 / pdf_1)
