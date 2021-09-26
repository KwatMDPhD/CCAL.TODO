from os import walk
from os.path import join


def list_child(pa):

    pa_ = []

    for di, di_, fi_ in walk(pa):

        pa_ += [join(di, na) for na in di_ + fi_]

    return sorted(pa_)
