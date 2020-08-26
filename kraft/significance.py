from numpy import asarray, sqrt, where
from scipy.stats import norm
from statsmodels.sandbox.stats.multicomp import multipletests


def get_moe(number_array, confidence=0.95):

    return norm.ppf(q=confidence) * number_array.std() / sqrt(number_array.size)


def get_p_value(number, random_number_, direction):

    if direction == "<":

        significant_n = (random_number_ <= number).sum()

    elif direction == ">":

        significant_n = (number <= random_number_).sum()

    return max(1, significant_n) / random_number_.size


def get_p_value_and_q_value(
    number_, random_number_, direction, multipletests_method="fdr_bh"
):

    if "<" in direction:

        left_p_value_ = asarray(
            tuple(get_p_value(number, random_number_, "<") for number in number_)
        )

        left_q_value_ = multipletests(left_p_value_, method=multipletests_method)[1]

    if ">" in direction:

        right_p_value_ = asarray(
            tuple(get_p_value(number, random_number_, ">") for number in number_)
        )

        right_q_value_ = multipletests(right_p_value_, method=multipletests_method)[1]

    if direction == "<":

        return left_p_value_, left_q_value_

    elif direction == ">":

        return right_p_value_, right_q_value_

    elif direction == "<>":

        return (
            where(left_p_value_ < right_p_value_, left_p_value_, right_p_value_),
            where(left_q_value_ < right_q_value_, left_q_value_, right_q_value_),
        )
