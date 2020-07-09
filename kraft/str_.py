from re import match, sub


def title(str_):

    title = ""

    for character, is_upper in zip(
        sub(r"_", " ", str_.title()), (character.isupper() for character in str_)
    ):

        if is_upper:

            character = character.upper()

        title += character

    for word in (
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
    ):

        title = sub(word.title(), word, title)

    return title


def untitle(str_):

    return sub(r"[ -]", "_", str_.lower())


def check_is_version(str_):

    return bool(match(r"^(0\.|[1-9]+\.){2}(0\.|[1-9]+)$", str_))


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


def make_unique(strs):

    uniques = []

    for str_ in strs:

        original = str_

        i = 2

        while str_ in uniques:

            str_ = "{}{}".format(original, i)

            i += 1

        uniques.append(str_)

    return uniques
