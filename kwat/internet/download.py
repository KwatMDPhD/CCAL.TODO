from os.path import exists
from re import search
from urllib.request import urlretrieve

from requests import get

from .get_name import get_name


def download(ur, pa, na=None, ov=True):

    if na is None:

        na = get_name(ur)

    pa = "{}{}".format(pa, na)

    if exists(pa):

        print("{} exists.".format(pa))

    if not exists(pa) or ov:

        print("{} => {}".format(ur, pa))

        if search(r"^ftp", ur):

            urlretrieve(ur, pa)

        else:

            with open(pa, mode="wb") as io:

                io.write(get(ur, allow_redirects=True).content)

    return pa
