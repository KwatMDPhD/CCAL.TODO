from os.path import abspath, expanduser, isdir


def get_absolute(pa):

    paa = abspath(expanduser(pa))

    if isdir(pa):

        paa += "/"

    return paa
