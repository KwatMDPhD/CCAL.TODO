from urllib.parse import unquote


def get_name(
    url,
):

    return unquote(url).split("/")[-1]
