from os.path import join

from kwat.path import make


def path(pa, gs):
    pa = join(pa, gs.lower(), "")

    make(pa)

    return pa, join(pa, "{}_{}_sa.tsv")
