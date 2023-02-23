def map_to(da, co, fu=None):
    an_cov = {}

    for an_, cov in zip(da.values, da.loc[:, co].values):
        for an in an_:
            if fu is None:
                an_cov[an] = cov

            else:
                for anr in fu(an):
                    an_cov[anr] = cov

    return an_cov
