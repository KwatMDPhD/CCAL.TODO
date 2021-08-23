from os import walk


def get_child(di):

    pa_ = []

    for di, di_, fi_ in walk(di):

        for na in di_:

            pa_.append("{}/{}/".format(di, na))

        for na in fi_:

            pa_.append("{}/{}".format(di, na))

    return sorted(pa_)
