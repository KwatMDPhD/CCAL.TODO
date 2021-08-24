def collapse(nu_fe_sa):

    print(nu_fe_sa.shape)

    nu_fe_sa = nu_fe_sa.groupby(level=0).median()

    print(nu_fe_sa.shape)

    return nu_fe_sa
