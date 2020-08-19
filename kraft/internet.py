from os import remove
from os.path import exists
from shutil import unpack_archive
from urllib.parse import unquote
from urllib.request import urlretrieve

from requests import get


def get_file_name(url):

    return unquote(url).split(sep="/")[-1]


def download(url, directory_path, file_name=None, overwrite=True):

    if file_name is None:

        file_name = get_file_name(url)

    file_path = "{}/{}".format(directory_path, file_name)

    if exists(file_path):

        print("{} exists.".format(file_path))

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

    unpack_archive(compressed_file_path, extract_dir=directory_path)

    remove(compressed_file_path)