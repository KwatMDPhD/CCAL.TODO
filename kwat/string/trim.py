def trim(st):
    n_ch = 24

    if n_ch < len(st):
        st = "{}...".format(st[:n_ch])

    return st
