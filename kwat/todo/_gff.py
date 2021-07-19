def get_gff3_attribute(attributes, field):

    for field_value in attributes.split(";"):

        (field_, value) = field_value.split("=", 1)

        if field_ == field:

            return value


from pandas import read_csv


def read_gff3(gff3_or_gff3_gz_file_path, only_type_to_keep=None):

    dataframe = read_csv(gff3_or_gff3_gz_file_path, "\t", comment="#")

    dataframe.columns = (
        "seqid",
        "source",
        "type",
        "start",
        "end",
        "score",
        "strand",
        "phase",
        "attributes",
    )

    if only_type_to_keep is not None:

        dataframe = dataframe[dataframe["type"] == only_type_to_keep]

    return dataframe
