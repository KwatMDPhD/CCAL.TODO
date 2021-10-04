def _print_shape(na, da_):

    print(na)

    for da in da_:

        print("\t{}".format(da.shape))


def sync(da_, ax):

    _print_shape("Before syncing:", da_)

    la_ = da_[0].axes[ax]

    for da in da_[1:]:

        la_ = la_.intersection(da.axes[ax])

    la_ = sorted(la_)

    das_ = [da.reindex(labels=la_, axis=ax) for da in da_]

    _print_shape("After:", das_)

    return das_
