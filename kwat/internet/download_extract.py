from os import remove
from shutil import unpack_archive

from .download import download


def download_extract(url, directory_path):

    compressed_file_path = download(url, directory_path)

    unpack_archive(compressed_file_path, extract_dir=directory_path)

    remove(compressed_file_path)
