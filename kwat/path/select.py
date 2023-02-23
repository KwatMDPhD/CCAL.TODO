from os import listdir
from os.path import join
from re import search


def select(pa, ig_=(r"^\.",), ke_=()):
    pa_ = []

    for na in sorted(listdir(path=pa)):
        if not any(search(ig, na) for ig in ig_) and (
            0 == len(ke_) or any(search(ke, na) for ke in ke_)
        ):
            pa_.append(join(pa, na))

    return pa_
