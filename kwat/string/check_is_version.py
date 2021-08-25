from re import search


def check_is_version(st):

    return bool(search(r"^(0\.|[1-9]+\.){2}(0\.|[1-9]+)$", st))
