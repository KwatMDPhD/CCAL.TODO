from cgi import parse_header
from os import remove
from re import sub
from shutil import unpack_archive
from urllib.request import urlopen, urlretrieve

from requests import get


def get_file_name(url):

    remote_file_info = urlopen(url).info()

    if "Content-Disposition" in remote_file_info:

        file_name = parse_header(remote_file_info["Content-Disposition"])[1]["filename"]

    else:

        file_name = sub("%2F", "/", url).split(sep="/")[-1]

    return file_name


def download(url, directory_path, file_name=None):

    if file_name is None:

        file_name = get_file_name(url)

    file_path = "{}/{}".format(directory_path, file_name)

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
