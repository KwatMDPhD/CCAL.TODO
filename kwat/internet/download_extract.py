from os import remove
from shutil import unpack_archive

from .download import download


def download_extract(ur, di):

    pa = download(ur, di)

    unpack_archive(pa, extract_dir=di)

    remove(pa)
