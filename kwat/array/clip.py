def clip(ar, st):

    me = ar.mean()

    st *= ar.std()

    return ar.clip(min=me - st, max=me + st)
