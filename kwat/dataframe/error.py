def error(da):

    for ax, la_ in enumerate(da.axes):

        na = "Axis {} ({})".format(ax, la_.name)

        assert not la_.isna().any(), "{} has Na.".format(na)

        assert not la_.has_duplicates, "{} is duplicated.".format(na)
