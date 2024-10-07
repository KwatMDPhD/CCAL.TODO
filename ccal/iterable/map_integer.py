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
