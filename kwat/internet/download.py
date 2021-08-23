from os.path import exists
from urllib.request import urlretrieve

from requests import get

from .get_name import get_name


def download(url, directory_path, name=None, overwrite=True):

    if name is None:

        name = get_name(url)

    file_path = "{}{}".format(directory_path, name)

    if exists(file_path):

        print("{} exists.".format(file_path))

    if not exists(file_path) or overwrite:

        print("{} => {}...".format(url, file_path))

        if url[:3] == "ftp":

            urlretrieve(url, file_path)

        else:

            with open(file_path, mode="wb") as io:

                io.write(get(url, allow_redirects=True).content)

    return file_path
