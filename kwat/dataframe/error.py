def error(da):

    for ax, la_ in enumerate(da.axes):

        for pr_, pr in [[la_.isna(), "Na"], [la_.duplicated(), "duplicates"]]:

            assert not pr_.any(), "Axis {} ({}) has {}:\n{}".format(
                ax, la_.name, pr, "\n".join(la_[pr_])
            )
