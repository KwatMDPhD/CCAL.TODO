from pandas import read_csv


def read_g2p(g2p_file_path):

    with open(g2p_file_path) as io:

        return {
            "header": [line for line in io if line.startswith("#")],
            "table": read_csv(g2p_file_path, sep="\t", comment="#"),
        }
