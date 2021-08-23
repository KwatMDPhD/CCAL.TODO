from os import listdir


def list(pa):

    return ["{}{}".format(pa, na) for na in sorted(listdir(pa)) if na[0] != "."]
