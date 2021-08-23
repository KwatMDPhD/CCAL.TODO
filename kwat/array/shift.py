def shift(ar, mi):

    if mi == "+1":

        return ar + 1

    elif mi == "0<":

        arp = 0 < ar

        if arp.any():

            mi = ar[arp].min()

        else:

            mi = 1

        print("Shifting the minimum to {}...".format(mi))

    return ar + mi - ar.min()
