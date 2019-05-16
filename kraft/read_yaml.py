from yaml import load


def read_yaml(yaml_file_path):

    with open(yaml_file_path) as io:

        return load(io)
