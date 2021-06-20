from os import listdir, mkdir, walk
from os.path import abspath, dirname, expanduser, isdir
from re import sub


def clean(na):

    nacl = sub(r"(?u)[^-\w.]", "_", na.strip().lower())

    print("{} => {}".format(na, nacl))

    return nacl


def get_absolute(pa):

    paab = abspath(expanduser(pa))

    if pa[-1] == "/":

        paab += "/"

    return paab


def list(pa):

    return ["{}{}".format(pa, na) for na in sorted(listdir(pa)) if na[0] != "."]


def get_child(di):

    pa_ = []

    for di, di_, fi_ in walk(di):

        for na in di_:

            pa_.append("{}/{}/".format(di, na))

        for na in fi_:

            pa_.append("{}/{}".format(di, na))

    return sorted(pa_)


def make(pa, pr=True):

    di = dirname(pa)

    di_ = []

    while di != "" and not isdir(di):

        di_.append(di)

        di = dirname(di)

    for di in di_[::-1]:

        mkdir(di)

        if pr:

            print("Made {}/".format(di))
