from re import sub


def standardize_file_name(file_name):

    return sub(r"(?u)[^-\w.]", "_", file_name.strip().lower().replace(" ", "_"))
