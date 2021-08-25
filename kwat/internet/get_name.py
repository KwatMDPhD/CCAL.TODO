from urllib.parse import unquote


def get_name(url):

    return unquote(url).split(sep="/")[-1]
