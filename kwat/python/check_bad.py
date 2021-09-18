from pandas import isna


def check_bad(an):

    if isinstance(an, str):

        return an.lower() in ["", "none", "na", "nan", "null"]

    else:

        return isna(an)
