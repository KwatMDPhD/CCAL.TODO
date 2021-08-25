from os.path import abspath, expanduser
from re import search


def get_absolute(pa):

    paa = abspath(expanduser(pa))

    if search(r"/$", pa):

        paa += "/"

    return paa
