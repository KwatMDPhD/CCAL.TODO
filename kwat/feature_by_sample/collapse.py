def collapse(nu_fe_sa):

    print("Before collapsing: {}".format(nu_fe_sa.shape))

    nu_fe_sa = nu_fe_sa.groupby(level=0).median()

    print("After: {}".format(nu_fe_sa.shape))

    return nu_fe_sa
