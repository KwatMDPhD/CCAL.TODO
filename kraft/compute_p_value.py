from numpy import nan


def compute_p_value(value, random_values, direction):

    assert value != nan

    if direction == "<":

        n_significant_random_value = (random_values <= value).sum()

    elif direction == ">":

        n_significant_random_value = (value <= random_values).sum()

    n_random_values = random_values.size

    return max(1, n_significant_random_value) / n_random_values
