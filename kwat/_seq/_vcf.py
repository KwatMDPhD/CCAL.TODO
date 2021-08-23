from os.path import isfile

import ravel
from pandas import read_csv, value_counts

from .shell import command


def get_variants_from_vcf_or_vcf_gz(
    vcf_or_vcf_gz_file_path,
    chromosome,
    _1_indexed_inclusive_start_position,
    _1_indexed_inclusive_end_position,
):

    if not isfile("{}.tbi".format(vcf_or_vcf_gz_file_path)):

        command("tabix {}".format(vcf_or_vcf_gz_file_path))

    return tuple(
        tuple(vcf_row.split("\t"))
        for vcf_row in command(
            "tabix {} {}:{}-{}".format(
                vcf_or_vcf_gz_file_path,
                chromosome,
                _1_indexed_inclusive_start_position,
                _1_indexed_inclusive_end_position,
            )
        )
        .stdout.strip()
        .splitlines()
        if vcf_row != ""
    )


def get_vcf_genotype(ref, alt, gt):

    genotypes = (ref, *alt.split(","))

    return tuple(genotypes[int(i)] for i in gt.replace("/", "|").split("|"))


def get_vcf_info_ann(info, key, n_ann=None):

    ann = get_vcf_info(info, "ANN")

    if ann is not None:

        i = VCF_ANN_KEYS.index(key)

        return tuple(
            ann_.split("|", i + 1)[i] for ann_ in ann.split(",", n_ann + 1)[:n_ann]
        )


def get_vcf_info(info, key):

    for info_ in info.split(";"):

        if "=" in info_:

            (info_key, info_value) = info_.split("=", 1)

            if info_key == key:

                return info_value


def get_vcf_sample_format(format_, key, sample):

    i = format_.split(":").index(key)

    return sample.split(":", i + 1)[i]


def make_variant_dict_from_vcf_row(vcf_row, n_info_ann=None):

    variant_dict = {
        column: vcf_row[i]
        for i, column in enumerate(VCF_COLUMNS[: VCF_COLUMNS.index("FILTER") + 1])
    }

    info_without_field = []

    for info in vcf_row[VCF_COLUMNS.index("INFO")].split(";"):

        if "=" in info:

            (info_field, info_value) = info.split("=", 1)

            if info_field == "ANN":

                variant_dict["ANN"] = {}

                for (ann_index, ann) in enumerate(
                    info_value.split(",", n_info_ann + 1)[:n_info_ann]
                ):

                    ann_values = ann.split("|")

                    variant_dict["ANN"][ann_index] = {
                        ann_field: ann_values[ann_field_index + 1]
                        for ann_field_index, ann_field in enumerate(VCF_ANN_KEYS[1:])
                    }

            else:

                variant_dict[info_field] = info_value

        else:

            info_without_field.append(info)

    if 0 < len(info_without_field):

        variant_dict["INFO_without_field"] = ";".join(info_without_field)

    vcf_column_format_index = VCF_COLUMNS.index("FORMAT")

    format_fields = vcf_row[vcf_column_format_index].split(":")

    variant_dict["sample"] = {}

    for (sample_index, sample) in enumerate(vcf_row[vcf_column_format_index + 1 :]):

        variant_dict["sample"][sample_index] = {
            format_field: sample_value
            for format_field, sample_value in zip(format_fields, sample.split(":"))
        }

    return variant_dict


def make_variant_n_from_vcf_file_path(vcf_file_path, use_only_pass=True):

    vcf = read_csv(vcf_file_path, "\t", comment="#", header=None, low_memory=False)

    filter_column = vcf.iloc[:, VCF_COLUMNS.index("FILTER")]

    if use_only_pass:

        is_pass = filter_column == "PASS"

        assert is_pass.any()

        vcf = vcf[is_pass]

    variant_n = value_counts(ravel(vcf.apply(make_variant_n_from_vcf_row, axis=1)))

    variant_n.index.name = "Variant"

    variant_n.name = "N"

    return variant_n


def make_variant_n_from_vcf_row(
    vcf_row,
):

    info = vcf_row[VCF_COLUMNS.index("INFO")]

    return set(
        "{} ({})".format(gene_name, effect)
        for gene_name, effect in zip(
            get_vcf_info_ann(info, "gene_name"), get_vcf_info_ann(info, "effect")
        )
    )


VCF_ANN_KEYS = (
    "ALT",
    "effect",
    "impact",
    "gene_name",
    "gene_id",
    "feature_type",
    "feature_id",
    "transcript_biotype",
    "rank",
    "hgvsc",
    "hgvsp",
    "cdna_position",
    "cds_position",
    "protein_position",
    "distance_to_feature",
    "error",
)
VCF_COLUMNS = ("CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT")


def count_gene_impacts_from_variant_dicts(variant_dicts, gene_name):

    impact_counts = {"HIGH": 0, "MODERATE": 0, "LOW": 0, "MODIFIER": 0}

    for variant_dict in variant_dicts:

        if variant_dict["gene_name"] == gene_name:

            impact_counts[variant_dict["impact"]] += 1

    return impact_counts


def get_variant_start_and_end_positions(pos, ref, alt):

    if len(ref) == len(alt):

        (start_position, end_position) = (pos, pos + len(alt) - 1)

    elif len(ref) < len(alt):

        (start_position, end_position) = (pos, pos + 1)

    else:

        (start_position, end_position) = (pos + 1, pos + len(ref) - len(alt))

    return (start_position, end_position)


def update_variant_dict(
    variant_dict,
):

    ref = variant_dict["REF"]

    alt = variant_dict["ALT"]

    (start_position, end_position) = get_variant_start_and_end_positions(
        int(variant_dict["POS"]), ref, alt
    )

    variant_dict["start_position"] = start_position

    variant_dict["end_position"] = end_position

    if "CAF" in variant_dict:

        variant_dict["population_allelic_frequencies"] = [
            float(caf_) for caf_ in variant_dict["CAF"].split(",")
        ]

    for sample_dict in variant_dict["sample"].values():

        if "GT" in sample_dict:

            sample_dict["genotype"] = get_vcf_genotype(ref, alt, sample_dict["GT"])

        if "AD" in sample_dict and "DP" in sample_dict:

            sample_dict["allelic_frequency"] = [
                int(ad_) / int(sample_dict["DP"])
                for ad_ in sample_dict["AD"].split(",")
            ]
