def shift(ar, mi):

    if mi == "+1":

        return ar + 1

    elif mi == "0<":

        arbo = 0 < ar

        if arbo.any():

            mi = ar[arbo].min()

        else:

            mi = 1

        print("Shifting the minimum to {}...".format(mi))

    return ar + mi - ar.min()
