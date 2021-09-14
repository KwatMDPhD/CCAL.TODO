def clip(nu___, st):

    me = nu___.mean()

    st *= nu___.std()

    return nu___.clip(min=me - st, max=me + st)
