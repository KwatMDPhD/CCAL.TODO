from numpy import asarray, where
from statsmodels.sandbox.stats.multicomp import multipletests

from .get_p_value import get_p_value


def get_p_values_and_q_values(
    values, random_values, direction, multipletests_method="fdr_bh"
):

    if "<" in direction:

        p_values_less = asarray(
            tuple(get_p_value(value, random_values, "<") for value in values)
        )

        q_values_less = multipletests(p_values_less, method=multipletests_method)[1]

    if ">" in direction:

        p_values_great = asarray(
            tuple(get_p_value(value, random_values, ">") for value in values)
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
