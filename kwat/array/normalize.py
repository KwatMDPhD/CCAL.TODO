from scipy.stats import rankdata


def normalize(ar, me, ra="average"):

    if me == "-0-":

        return (ar - ar.mean()) / ar.std()

    elif me == "0-1":

        mi = ar.min()

        return (ar - mi) / (ar.max() - mi)

    elif me == "sum":

        return ar / ar.sum()

    elif me == "rank":

        return rankdata(ar, method=ra).reshape(ar.shape)
