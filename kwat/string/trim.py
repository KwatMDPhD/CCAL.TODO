def trim(text):

    n_ch = 24

    if n_ch < len(text):

        text = "{}...".format(text[:n_ch])

    return text
