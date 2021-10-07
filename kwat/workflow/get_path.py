from os import readlink
from os.path import dirname, islink, join

from ..path import get_absolute


def get_path(se):

    se = get_absolute(se)

    if islink(se):

        se = readlink(se)

    par = dirname(dirname(se))

    pai = join(par, "input", "")

    pac = join(par, "code", "")

    pao = join(par, "output", "")

    return par, pai, pac, pao
