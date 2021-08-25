from os import listdir
from re import search


def list(pa):

    return [
        "{}{}".format(pa, na) for na in sorted(listdir(path=pa)) if search(r"^[^.]", na)
    ]
