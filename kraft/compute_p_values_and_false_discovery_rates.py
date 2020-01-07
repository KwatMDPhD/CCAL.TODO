from numpy import asarray, where
from statsmodels.sandbox.stats.multicomp import multipletests

from .compute_p_value import compute_p_value


def compute_p_values_and_false_discovery_rates(
    values, random_values, direction, method="fdr_bh"
):

    if "<" in direction:

        p_values_less = asarray(
            tuple(compute_p_value(value, random_values, "<") for value in values)
        )

        false_discovery_rates_less = multipletests(p_values_less, method=method)[1]

    if ">" in direction:

        p_values_great = asarray(
            tuple(compute_p_value(value, random_values, ">") for value in values)
        )

        false_discovery_rates_great = multipletests(p_values_great, method=method)[1]

    if direction == "<>":

        p_values = where(p_values_less < p_values_great, p_values_less, p_values_great)

        false_discovery_rates = where(
            false_discovery_rates_less < false_discovery_rates_great,
            false_discovery_rates_less,
            false_discovery_rates_great,
        )

    elif direction == "<":

        p_values = p_values_less

        false_discovery_rates = false_discovery_rates_less

    elif direction == ">":

        p_values = p_values_great

        false_discovery_rates = false_discovery_rates_great

    return p_values, false_discovery_rates
