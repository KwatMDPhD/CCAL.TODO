from os import mkdir
from os.path import dirname, isdir


def make(pa, pr=True):
    di_ = []

    di = dirname(pa)

    while di != "" and not isdir(di):
        di_.append(di)

        di = dirname(di)

    for di in di_[::-1]:
        mkdir(di)

        if pr:
            print("Made {}.".format(di))
