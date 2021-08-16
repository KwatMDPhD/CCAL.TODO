from re import sub


def title(st):

    ti = ""

    for bo, ch in zip((ch.isupper() for ch in st), sub(r"_", " ", st.title())):

        if bo:

            ch = ch.upper()

        ti += ch

    for wo in [
        r" a ",
        r" an ",
        r" the ",
        r" and ",
        r" but ",
        r" or ",
        r" for ",
        r" nor ",
        r" on ",
        r" at ",
        r" to ",
        r" from ",
        r" of ",
        r" vs ",
        r"'m",
        r"'s",
        r"'re",
    ]:

        ti = sub(wo.title(), wo, ti)

    return ti
