# TODO: use regular expression


def title(str_):

    title = ""

    for character, is_upper in zip(
        str_.title().replace("_", " "), (character.isupper() for character in str_)
    ):

        if is_upper:

            character = character.upper()

        title += character

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

    return str_.lower().replace(" ", "_").replace("-", "_")


def check_is_version(str_):

    split = str_.split(sep=".")

    return len(split) == 3 and all(str_.isnumeric() for str_ in split)


def skip_quote_and_split(str_, separator):

    splits = []

    part = ""

    for split in str_.split(sep=separator):

        if '"' in split:

            if part == "":

                part = split

            else:

                part += separator + split

                splits.append(part)

                part = ""

        else:

            if part == "":

                splits.append(split)

            else:

                part += split

    if part != "":

        splits.append(part)

    return splits
