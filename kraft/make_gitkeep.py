from os.path import join


def make_gitkeep(directory_path):

    gitkeep_file_path = join(directory_path, ".gitkeep")

    open(gitkeep_file_path, mode="w").close()
