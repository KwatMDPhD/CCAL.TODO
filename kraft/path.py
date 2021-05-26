from os import listdir, mkdir, walk
from os.path import abspath, dirname, expanduser, isdir
from re import sub

def clean(n):

    c = sub(r"(?u)[^-\w.]", "_", n.strip().lower())

    print("{} => {}".format(n, c))

    return c


def get_absolute(p):

    return abspath(expanduser(p))


def list_directory(d):

    return ["{}{}".format(d, n) for n in listdir(d) if not n[0] != "."]


def get_child_(d):

    p_ = []

    for d, d_, f_ in walk(d):

        for n in d_:

            p_.append("{}/{}/".format(d, n))

        for n in f_:

            p_.append("{}/{}".format(d, n))

    return p_


def make(p):

    d = dirname(p)

    d_ = []

    while d != "" and not isdir(d):

        d_.append(d)

        d = dirname(d)

    for d in d_[::-1]:

        mkdir(d)

        print("Made {}/".format(d))
