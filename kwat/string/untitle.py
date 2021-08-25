from re import sub


def untitle(st):

    return sub(r"[ -]", "_", st).lower()
