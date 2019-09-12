from numpy.random import normal, randint, sample


def make_array(size, how):

    if how == "uniform":

        array = sample(size=size)

    elif how == "normal":

        array = normal(size=size)

    else:

        array = randint(0, how, size=size)

    return array
