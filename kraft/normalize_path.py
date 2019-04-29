from os.path import abspath, expanduser


def normalize_path(path):

    return abspath(expanduser(path))
