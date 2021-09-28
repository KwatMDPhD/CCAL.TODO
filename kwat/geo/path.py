from os.path import join

from kwat.path import make


def path(pa, gs):

    pag = join(pa, gs.lower(), "")

    make(pag)

    return pag, join(pag, "{}_{}_sa.tsv")
