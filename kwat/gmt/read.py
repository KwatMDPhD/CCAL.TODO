from ._read import _read


def read(pa_):

    se_ge = {}

    for pa in pa_:

        se_ge.update(_read(pa))

    return se_ge
