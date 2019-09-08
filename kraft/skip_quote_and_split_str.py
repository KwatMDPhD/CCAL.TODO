def skip_quote_and_split_str(str_, separator):

    splits = []

    part = ""

    for str_split in str_.split(sep=separator):

        if '"' in str_split:

            if part == "":

                part = str_split

            else:

                part += str_split

                splits.append(part)

                part = ""

        else:

            if part == "":

                splits.append(str_split)

            else:

                part += str_split

    if part != "":

        splits.append(part)

    return splits
