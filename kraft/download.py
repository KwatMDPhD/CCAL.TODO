from os.path import basename
from urllib.parse import urlsplit
from urllib.request import urlretrieve


def download(url, directory_path):

    file_path = "{}/{}".format(directory_path, basename(urlsplit(url).path))

    print("{} ==> {}...".format(url, file_path))

    urlretrieve(url, file_path)

    return file_path
