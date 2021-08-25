from os.path import basename
from urllib.parse import unquote


def get_name(ur):

    return basename(unquote(ur))
