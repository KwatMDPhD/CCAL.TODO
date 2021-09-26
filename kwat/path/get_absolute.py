from os.path import abspath, expanduser


def get_absolute(pa):

    return abspath(expanduser(pa))
