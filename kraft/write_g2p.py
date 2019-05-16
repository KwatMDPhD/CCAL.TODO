def write_g2p(g2p, g2p_file_path):

    if not all(line.startswith("#") for line in g2p["header"]):

        raise ValueError("Each header line should start with a '#'.")

    with open(g2p_file_path, "w") as io:

        if 0 < len(g2p["header"]):

            io.write("\n".join(g2p["header"]) + "\n")

        g2p["table"].to_csv(io, sep="\t", index=None)
