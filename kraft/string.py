from re import match, sub


def title(s):

    t = ""

    for isu, c in zip((c.isupper() for c in s), sub(r"_", " ", s.title())):

        if isu:

            c = c.upper()

        t += c

    for w in [
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

        t = sub(w.title(), w, t)

    return t


def untitle(s):

    return sub(r"[ -]", "_", s.lower())


def check_is_version(s):

    return bool(match(r"^(0\.|[1-9]+\.){2}(0\.|[1-9]+)$", s))


def skip_quote_and_split(str_, separator=" "):

    splits = []

    quote = ""

    for split in str_.split(sep=separator):

        if '"' in split:

            if quote == "":

                quote = split

            else:

                quote += separator + split

                splits.append(quote)

                quote = ""

        else:

            if quote == "":

                splits.append(split)

            else:

                quote += split

    if quote != "":

        splits.append(quote)

    return splits


def make_unique(s_):

    u_ = []

    for s in s_:

        o = s

        i = 2

        while s in u_:

            s = "{}{}".format(o, i)

            i += 1

        u_.append(s)

    return u_
