from os import listdir, mkdir, walk
from os.path import abspath, dirname, expanduser, isdir
from re import sub


def make(p):

    dp = dirname(p)

    mdp_ = []

    while dp != "" and not isdir(dp):

        mdp_.append(dp)

        dp = dirname(dp)

    for dp in mdp_[::-1]:

        mkdir(dp)

        print("{}/".format(dp))


def get_cp(dp):

    cp_ = []

    for _dp, dn_, fn_ in walk(dp):

        for n in dn_:

            cp_.append("{}/{}/".format(_dp, n))

        for n in fn_:

            cp_.append("{}/{}".format(_dp, n))

    return cp_


def clean(n):

    cn = sub(r"(?u)[^-\w.]", "_", n.strip().lower())

    print("{} => {}".format(n, cn))

    return cn


def make_absolute(p):

    return abspath(expanduser(p))


def list_absolute(dp):

    return [
        "{}{}".format(dp, n) for n in listdir(path=dp)
    ]
