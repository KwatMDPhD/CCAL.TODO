from os.path import exists, join
from re import search
from urllib.request import urlretrieve

from requests import get

from .get_name import get_name


def download(ur, pa, na=None, ov=True):
    if na is None:
        na = get_name(ur)

    pan = join(pa, na)

    if exists(pan):
        print("{} exists.".format(pan))

    if not exists(pan) or ov:
        print("{} => {}".format(ur, pan))

        if search(r"^ftp", ur):
            urlretrieve(ur, pan)

        else:
            with open(pan, mode="wb") as io:
                io.write(get(ur, allow_redirects=True).content)

    return pan
