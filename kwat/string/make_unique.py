def make_unique(st_):

    un_ = []

    for st in st_:

        stun = st

        it = 2

        while stun in un_:

            stun = "{}{}".format(st, it)

            it += 1

        un_.append(stun)

    return un_
