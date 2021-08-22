def map_to(da, la, fu=None):

    ke_va = {}

    for ke_, va in zip(da.values, da[la].values):

        for ke in ke_:

            if fu is None:

                ke_va[ke] = va

            else:

                for ke in fu(ke):

                    ke_va[ke] = va

    return ke_va
