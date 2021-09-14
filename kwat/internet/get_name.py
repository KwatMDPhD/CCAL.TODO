from urllib.parse import unquote


def get_name(ur):

    return unquote(ur).split(sep="/")[-1]
