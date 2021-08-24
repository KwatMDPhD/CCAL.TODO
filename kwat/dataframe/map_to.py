def map_to(da, co, fu=None):

    ro_co = {}

    for ro, cov in zip(da.values, da.loc[:, co].values):

        for rov in ro:

            if fu is None:

                ro_co[rov] = cov

            else:

                for rovf in fu(rov):

                    ro_co[rovf] = cov

    return ro_co
