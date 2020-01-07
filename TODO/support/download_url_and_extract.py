from os import remove
from os.path import splitext
from shutil import unpack_archive

from .download_url import download_url


def download_url_and_extract(url, directory_path):

    compressed_file_path = download_url(url, directory_path)

    unpack_archive(compressed_file_path, extract_dir=splitext(compressed_file_path)[0])

    remove(compressed_file_path)
