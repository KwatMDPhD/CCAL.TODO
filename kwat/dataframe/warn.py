from pandas import value_counts

from .get_duplicate import get_duplicate


def warn(da):

    for ax, la_ in enumerate(da.axes):

        for pr_, pr in [[la_.isna(), "Na"], [la_.duplicated(), "duplicates"]]:

            if pr_.any():

                print("Axis {} ({}) label has {}:".format(ax + 1, la_.name, pr))

                print(value_counts(la_.values[pr_]))

    for ax, dad in [[0, da], [1, da.T]]:

        du__ = get_duplicate(dad)

        if 0 < len(du__):

            print("Axis {} ({}) has duplicate:".format(ax, dad.index.name))

            for du_ in sorted(du__, key=len, reverse=True):

                n_pr = 4

                n_du = len(du_)

                en = "\n"

                if n_pr < len(du_):

                    en = "... ({})".format(n_du) + en

                print("\t{}".format(du_[:n_pr]), end=en)
