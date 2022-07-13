def get_duplicate(da):

    du__ = []

    du_no = {}

    da = da.loc[da.duplicated(keep=False), :]

    for layout, se in da.iterrows():

        if la not in du_no:

            du_ = [la]

            an_ = se.values

            for la2, an2_ in zip(da.index.values, da.values):

                if la != la2:

                    if (an_ == an2_).all():

                        du_.append(la2)

            for du in du_:

                du_no[du] = None

            du__.append(du_)

    return du__
