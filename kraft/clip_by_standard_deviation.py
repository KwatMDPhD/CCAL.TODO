def clip_array_by_standard_deviation(array, standard_deviation):

    array_mean = array.mean()

    array_good_interval = array.std() * standard_deviation

    return array.clip(
        min=array_mean - array_good_interval, max=array_mean + array_good_interval,
    )
