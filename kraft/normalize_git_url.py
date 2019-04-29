def normalize_git_url(git_url):

    for str_ in ("/", ".git"):

        if git_url.endswith(str_):

            git_url = git_url[: -len(str_)]

    return git_url
