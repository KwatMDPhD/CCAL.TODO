def map_to_column(da, co, fu=None):

    ke_va = {}

    for va, ke_ in zip(da.pop(co).to_numpy(), da.to_numpy()):

        for ke in ke_:

            if fu is None:

                ke_va[ke] = va

            else:

                for ke in fu(ke):

                    ke_va[ke] = va

    return ke_va
