from numpy import isnan, nan


def get_p_value(value, random_values, direction):

    if isnan(value):

        return nan

    if direction == "<":

        n_significant = (random_values <= value).sum()

    elif direction == ">":

        n_significant = (value <= random_values).sum()

    return max(1, n_significant) / random_values.size
