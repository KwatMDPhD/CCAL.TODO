def make_unique(st_):
    un_ = []

    for st in st_:
        un = st

        it = 2

        while un in un_:
            un = "{}{}".format(st, it)

            it += 1

        un_.append(un)

    return un_
