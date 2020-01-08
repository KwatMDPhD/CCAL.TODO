def clip_by_standard_deviation(array, standard_deviation):

    mean = array.mean()

    margin = array.std() * standard_deviation

    return array.clip(min=mean - margin, max=mean + margin)
