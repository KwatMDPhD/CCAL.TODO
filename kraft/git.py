from os.path import join
from subprocess import CalledProcessError

from .is_str_version import is_str_version
from .run_command import run_command


def get_git_versions():

    tags = run_command("git tag --list").stdout.strip().split(sep="\n")

    versions = [tag for tag in tags if is_str_version(tag)]

    versions.sort(key=lambda iii: tuple(int(i) for i in iii.split(sep=".")))

    return versions


def is_in_git_repository():

    try:

        run_command("git status")

        return True

    except CalledProcessError:

        return False


def make_gitkeep(directory_path):

    open(join(directory_path, ".gitkeep"), mode="w").close()


def normalize_git_url(git_url):

    for str_ in ("/", ".git"):

        if git_url.endswith(str_):

            git_url = git_url[: -len(str_)]

    return git_url
