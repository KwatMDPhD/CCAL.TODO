from os import walk


def get_child(di):

    pa_ = []

    for di, di_, fi_ in walk(di):

        te = "{}/{{}}".format(di)

        for na in di_:

            pa_.append(te.format(na) + "/")

        for na in fi_:

            pa_.append(te.format(na))

    return sorted(pa_)
