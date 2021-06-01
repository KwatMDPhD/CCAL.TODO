from re import match, sub


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


def untitle(st):

    return sub(r"[ -]", "_", st.lower())


def check_is_version(st):

    return bool(match(r"^(0\.|[1-9]+\.){2}(0\.|[1-9]+)$", st))


def split(st, se=" "):

    sp_ = []

    qu = ""

    for sp in st.split(se):

        if '"' in sp or "'" in sp:

            if qu == "":

                qu = sp

            else:

                qu += se + sp

                sp_.append(qu)

                qu = ""

        else:

            if qu == "":

                sp_.append(sp)

            else:

                qu += sp

    if qu != "":

        sp_.append(qu)

    return sp_


def make_unique(st_):

    un_ = []

    for st in st_:

        stun = st

        it = 2

        while stun in un_:

            stun = "{}{}".format(st, it)

            it += 1

        un_.append(stun)

    return un_
