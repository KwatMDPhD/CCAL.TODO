from re import sub


def clean(na):

    nac = sub(r"[^\w.]", "_", na.strip().lower())

    print("{} => {}".format(na, nac))

    return nac
