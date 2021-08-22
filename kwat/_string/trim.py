def trim(text):

    n_ch = 25

    if n_ch < len(text):

        text = "{}...".format(text[:n_ch])

    return text
