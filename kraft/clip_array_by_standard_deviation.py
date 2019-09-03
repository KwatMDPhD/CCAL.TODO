from .check_array_for_bad import check_array_for_bad


def clip_array_by_standard_deviation(array, standard_deviation, raise_for_bad=True):

    is_good = ~check_array_for_bad(array, raise_for_bad=raise_for_bad)

    array_copy = array.copy()

    if not is_good.any():

        return array_copy

    array_copy_good = array_copy[is_good]

    array_copy_good_mean = array_copy_good.mean()

    array_copy_good_interval = array_copy_good.std() * standard_deviation

    array_copy[is_good] = array_copy[is_good].clip(
        min=array_copy_good_mean - array_copy_good_interval,
        max=array_copy_good_mean + array_copy_good_interval,
    )

    return array_copy
