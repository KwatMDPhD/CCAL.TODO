def sync(da_, ax):

    la_ = da_[0].axes[ax]

    for da in da_[1:]:

        la_ = la_.intersection(da.axes[ax])

    la_ = sorted(la_)

    return [da.reindex(la_, axis=ax) for da in da_]
