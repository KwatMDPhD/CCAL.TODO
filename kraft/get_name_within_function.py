from inspect import stack


def get_name_within_function():

    return stack()[1][3]
