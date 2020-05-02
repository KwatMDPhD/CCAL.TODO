from os.path import basename, exists
from urllib.parse import urlsplit
from urllib.request import urlretrieve

from requests import get


def download(url, directory_path, overwrite=True):

    file_path = "{}/{}".format(directory_path, basename(urlsplit(url).path))

    if not exists(file_path) or overwrite:

        print("{} ==> {}...".format(url, file_path))

        if url.startswith("ftp"):

            urlretrieve(url, file_path)

        else:

            with open(file_path, mode="wb") as io:

                io.write(get(url, allow_redirects=True).content)

    return file_path
