from numpy import asarray, isnan, sqrt, where
from scipy.stats import norm
from statsmodels.sandbox.stats.multicomp import multipletests


def get_moe(numbers, confidence=0.95):

    return norm.ppf(q=confidence) * numbers.std() / sqrt(numbers.size)


def get_p_value(number, random_numbers, direction):

    assert not isnan(number)

    if direction == "<":

        n_significant = (random_numbers <= number).sum()

    elif direction == ">":

        n_significant = (number <= random_numbers).sum()

    return max(1, n_significant) / random_numbers.size


def get_p_values_and_q_values(
    numbers, random_numbers, direction, multipletests_method="fdr_bh"
):

    if "<" in direction:

        p_values_less = asarray(
            tuple(get_p_value(number, random_numbers, "<") for number in numbers)
        )

        q_values_less = multipletests(p_values_less, method=multipletests_method)[1]

    if ">" in direction:

        p_values_great = asarray(
            tuple(get_p_value(number, random_numbers, ">") for number in numbers)
        )

        q_values_great = multipletests(p_values_great, method=multipletests_method)[1]

    if direction == "<":

        return p_values_less, q_values_less

    elif direction == ">":

        return p_values_great, q_values_great

    elif direction == "<>":

        return (
            where(p_values_less < p_values_great, p_values_less, p_values_great),
            where(q_values_less < q_values_great, q_values_less, q_values_great),
        )
