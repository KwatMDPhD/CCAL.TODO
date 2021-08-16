def clip(ar, st):

    me = ar.mean()

    st *= ar.std()

    return ar.clip(me - st, me + st)
