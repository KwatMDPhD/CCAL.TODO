from os import readlink
from os.path import dirname, islink

from ..path import get_absolute


def get_root(se):

    if islink(se):

        se = readlink(se)

    return dirname(dirname(get_absolute(se)))
