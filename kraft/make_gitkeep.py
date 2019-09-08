from os.path import join


def make_gitkeep(directory_path):

    open(join(directory_path, ".gitkeep"), mode="w").close()
