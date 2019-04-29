def is_str_version(str_):

    str_split = str_.split(sep=".")

    return "." in str_ and len(str_split) == 3 and all(i.isnumeric() for i in str_split)
