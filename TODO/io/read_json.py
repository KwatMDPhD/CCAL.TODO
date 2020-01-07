from json import load


def read_json(json_file_path):

    with open(json_file_path) as io:

        return load(io)
