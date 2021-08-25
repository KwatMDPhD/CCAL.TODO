from os.path import abspath, expanduser


def get_absolute(pa):

    paa = abspath(expanduser(pa))

    if pa[-1] == "/":

        paa += "/"

    return paa
