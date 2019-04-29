from re import sub


def normalize_file_name(file_name):

    return sub(r"(?u)[^-\w.]", "_", file_name.strip().replace(" ", "_"))
