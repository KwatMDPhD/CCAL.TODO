from re import match


def check_is_version(st):

    return bool(match(r"^(0\.|[1-9]+\.){2}(0\.|[1-9]+)$", st))
