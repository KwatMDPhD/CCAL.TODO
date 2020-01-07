from numpy import asarray, unique

from .is_array_bad import is_array_bad


def guess_data_type(data):

    data = asarray(data)

    data_good = data[~is_array_bad(data, raise_if_bad=False)]

    if all(float(n).is_integer() for n in data_good):

        n_good_unique = unique(data_good).size

        if n_good_unique == 2:

            return "binary"

        elif n_good_unique <= 24:

            return "categorical"

    return "continuous"
