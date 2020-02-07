from numpy import absolute, concatenate, cumsum, inf

from .compute_kullback_leibler_divergence import compute_kullback_leibler_divergence


def compute_pdf_and_pdf_reference_context(
    grid, pdf, pdf_reference, multiply_distance_from_reference_argmax
):

    center = pdf_reference.argmax()

    left_kl = compute_kullback_leibler_divergence(pdf[:center], pdf_reference[:center])

    right_kl = compute_kullback_leibler_divergence(pdf[center:], pdf_reference[center:])

    left_kl[left_kl == inf] = 0

    right_kl[right_kl == inf] = 0

    left_context = -cumsum((left_kl / left_kl.sum())[::-1])[::-1]

    right_context = cumsum(right_kl / right_kl.sum())

    left_context *= left_kl.sum() / left_kl.size

    right_context *= right_kl.sum() / right_kl.size

    context = concatenate((left_context, right_context))

    if multiply_distance_from_reference_argmax:

        context *= absolute(grid - grid[center])

    return context
