from re import sub


def title(st):
    ti = ""

    for up, ch in zip((ch.isupper() for ch in st), sub(r"_", " ", st).title()):
        if up:
            ch = ch.upper()

        ti += ch

    for lo in [
        " a ",
        " an ",
        " the ",
        " and ",
        " but ",
        " or ",
        " for ",
        " nor ",
        " on ",
        " at ",
        " to ",
        " from ",
        " of ",
        " vs ",
        "'m",
        "'s",
        "'re",
    ]:
        ti = sub(r"{}".format(lo.title()), lo, ti)

    return ti
