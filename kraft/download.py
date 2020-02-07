from os.path import basename
from urllib.parse import urlsplit

from requests import get


def download(url, directory_path):

    file_path = "{}/{}".format(directory_path, basename(urlsplit(url).path))

    print("{} =(download)=> {}...".format(url, file_path))

    with open(file_path, mode="wb") as io:

        io.write(get(url, allow_redirects=True).content)

    return file_path
