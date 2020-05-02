from json import dump


def write_json(json_file_path, dict_, indent=2):

    with open(json_file_path, mode="w") as io:

        dump(dict_, io, indent=indent)
