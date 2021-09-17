from os import remove
from shutil import unpack_archive

from .download import download


def download_and_extract(ur, pa):

    pa = download(ur, pa)

    unpack_archive(pa, extract_dir=pa)

    remove(pa)
