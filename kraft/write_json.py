from json import dump


def write_json(dict_, json_file_path, indent=2):

    with open(json_file_path, mode="w") as io:

        dump(dict_, io, indent=indent)
