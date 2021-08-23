from re import sub


def clean(na):

    nacl = sub(r"(?u)[^-\w.]", "_", na.strip().lower())

    print("{} => {}".format(na, nacl))

    return nacl
