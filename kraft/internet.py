from os import remove
from os.path import basename, exists, splitext
from shutil import unpack_archive
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


def download_and_extract(url, directory_path):

    compressed_file_path = download(url, directory_path)

    unpack_archive(compressed_file_path, extract_dir=splitext(compressed_file_path)[0])

    remove(compressed_file_path)
