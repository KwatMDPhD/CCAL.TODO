from numpy import nan


def compute_p_value(value, random_values, direction):

    assert value != nan

    if direction == "<":

        n_significant_random_value = (random_values <= value).sum()

    elif direction == ">":

        n_significant_random_value = (value <= random_values).sum()

    return max(1, n_significant_random_value) / random_values.size
