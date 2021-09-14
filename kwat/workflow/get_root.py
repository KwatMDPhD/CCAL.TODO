from os import readlink
from os.path import dirname

from ..path import get_absolute


def get_root(se):

    return get_absolute(dirname(dirname(readlink(se))))
