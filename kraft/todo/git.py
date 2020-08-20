from subprocess import CalledProcessError

from .shell import command
from .str_ import is_version


def get_git_versions():

    tags = command("git tag --list").stdout.strip().split(sep="\n")

    versions = [tag for tag in tags if is_version(tag)]

    def function(iii):

        return tuple(int(i) for i in iii.split(sep="."))

    versions.sort(key=function)

    return versions


def is_in_git_repository():

    try:

        command("git status")

        return True

    except CalledProcessError:

        return False


def make_gitkeep(directory_path):

    open("{}.gitkeep".format(directory_path), mode="w").close()


def normalize_git_url(git_url):

    for str_ in ("/", ".git"):

        if git_url.endswith(str_):

            git_url = git_url[: -len(str_)]

    return git_url
