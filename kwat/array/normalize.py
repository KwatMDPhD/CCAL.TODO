from scipy.stats import rankdata


def normalize(nu___, me, ra="average"):

    if me == "-0-":

        return (nu___ - nu___.mean()) / nu___.std()

    elif me == "0-1":

        mi = nu___.min()

        return (nu___ - mi) / (nu___.max() - mi)

    elif me == "sum":

        return nu___ / nu___.sum()

    elif me == "rank":

        return rankdata(nu___, method=ra).reshape(nu___.shape)
