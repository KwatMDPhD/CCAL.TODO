from os.path import abspath, expanduser


def get_absolute(pa):

    paab = abspath(expanduser(pa))

    if pa[-1] == "/":

        paab += "/"

    return paab
