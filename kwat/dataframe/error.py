def error(da):

    for ie, la_ in enumerate(da.axes):

        ar_ = ie + 1, la_.name

        assert not la_.isna().any(), "Dimension {} ({}) has Na.".format(*ar_)

        assert not la_.has_duplicates, "Dimension {} ({}) is duplicated.".format(*ar_)
