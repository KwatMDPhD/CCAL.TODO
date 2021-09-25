from os import walk
from os.path import join


def list_child(pa):

    pa_ = []

    for di, di_, fi_ in walk(pa):

        te = join(di, "{}")

        for na in di_:

            pa_.append(te.format(na) + "/")

        for na in fi_:

            pa_.append(te.format(na))

    return sorted(pa_)
