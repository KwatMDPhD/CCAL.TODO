from os import listdir
from os.path import isdir
from re import search


def list_directory(pa):

    pa_ = []

    for na in sorted(listdir(path=pa)):

        if search(r"^[^.]", na):

            pan = "{}{}".format(pa, na)

            if isdir(pan):

                pan += "/"

            pa_.append(pan)

    return pa_
