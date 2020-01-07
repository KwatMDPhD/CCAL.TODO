from re import sub


def normalize_name(name):

    return sub(r"(?u)[^-\w.]", "_", name.strip().lower().replace(" ", "_"))
