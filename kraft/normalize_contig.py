def normalize_contig(contig, format_):

    contig = str(contig)

    if format_ == "chr":

        if not contig.startswith("chr"):

            contig = f"chr{contig}".replace("MT", "M")

    elif format_ == "":

        if contig.startswith("chr"):

            contig = contig[3:].replace("M", "MT")

    return contig
