from numpy import asarray


def check_is_iterable(an_):

    try:

        iter(an_)

        return True

    except:

        return False


def flatten(an_, ty_=(tuple, list, set)):

    an_ = list(an_)

    ie = 0

    while ie < len(an_):

        while isinstance(an_[ie], ty_):

            if len(an_[ie]) == 0:

                an_.pop(ie)

                ie -= 1

                break

            else:

                an_[ie : ie + 1] = an_[ie]

        ie += 1

    return an_


def map_integer(an_):

    an_it = {}

    it_an = {}

    it = 1

    for an in an_:

        if an not in an_it:

            an_it[an] = it

            it_an[it] = an

            it += 1

    return an_it, it_an


def check_is_in(an1_, an2_):

    an2_no = {an2: None for an2 in an2_}

    return asarray([an1 in an2_no for an1 in an1_], bool)
