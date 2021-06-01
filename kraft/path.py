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

    return ["{}{}".format(pa, na) for na in listdir(pa) if na[0] != "."]


def get_child(di):

    pa_ = []

    for di, di_, fi_ in walk(di):

        for na in di_:

            pa_.append("{}/{}/".format(di, na))

        for n in fi_:

            pa_.append("{}/{}".format(di, na))

    return pa_


def make(pa):

    di = dirname(pa)

    di_ = []

    while di != "" and not isdir(di):

        di_.append(di)

        di = dirname(di)

    for di in di_[::-1]:

        mkdir(di)

        print("Made {}/".format(di))
