from re import sub


def clean(na):

    nac = sub(r"(?u)[^-\w.]", "_", na.strip().lower())

    print("{} => {}".format(na, nac))

    return nac
