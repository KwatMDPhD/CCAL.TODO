from os import listdir
from os.path import isdir
from re import search


def list_directory(pa, ig_=(r"^\.",), ke_=()):

    pa_ = []

    for na in sorted(listdir(path=pa)):

        if 0 < len(ig_) and any(search(ig, na) for ig in ig_):

            continue

        if 0 < len(ke_) and not any(search(ke, na) for ke in ke_):

            continue

        paa = "{}{}".format(pa, na)

        if isdir(paa):

            paa += "/"

        pa_.append(paa)

    return pa_
