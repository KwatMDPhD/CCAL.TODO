from os import remove
from os.path import splitext
from shutil import unpack_archive

from .download import download


def download_extract(url, directory_path):

    compressed_file_path = download(url, directory_path)

    unpack_archive(compressed_file_path, extract_dir=splitext(compressed_file_path)[0])

    remove(compressed_file_path)
