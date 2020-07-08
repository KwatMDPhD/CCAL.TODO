def title(str_):

    title = ""

    # TODO: use regular expression
    for character, is_upper in zip(
        str_.title().replace("_", " "), (character.isupper() for character in str_)
    ):

        if is_upper:

            character = character.upper()

        title += character

    # TODO: use regular expression
    for word in (
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
    ):

        title = title.replace(word.title(), word)

    return title


def untitle(str_):

    # TODO: use regular expression
    return str_.lower().replace(" ", "_").replace("-", "_")


def check_is_version(str_):

    split = str_.split(sep=".")

    # TODO: use regular expression
    return len(split) == 3 and all(str_.isnumeric() for str_ in split)


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
