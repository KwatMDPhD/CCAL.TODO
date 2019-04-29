from collections import defaultdict
from gzip import open as gzip_open
from pickle import dump, load

from tables import (
    Filters,
    Float32Col,
    HDF5ExtError,
    Int32Col,
    IsDescription,
    StringCol,
    open_file,
)

from .get_vcf_info import get_vcf_info
from .get_vcf_info_ann import get_vcf_info_ann
from .get_vcf_sample_format import get_vcf_sample_format
from .make_variant_dict_consistent import make_variant_dict_consistent
from .read_where_and_map_column_name_on_hdf5_table import (
    read_where_and_map_column_name_on_hdf5_table,
)
from .update_variant_dict import update_variant_dict


class _VariantDescription(IsDescription):

    CHROM = StringCol(256)

    POS = Int32Col()

    ID = StringCol(256)

    REF = StringCol(256)

    ALT = StringCol(256)

    QUAL = Float32Col()

    CAF = StringCol(256)

    CLNDISDB = StringCol(256)

    CLNDN = StringCol(256)

    CLNSIG = StringCol(256)

    CLNREVSTAT = StringCol(256)

    CLNVI = StringCol(256)

    effect = StringCol(256)

    impact = StringCol(256)

    gene_name = StringCol(256)

    GT = StringCol(256)


class VariantHDF5:
    def __init__(self, vcf_gz_file_path, reset=False):

        self.vcf_gz_file_path = vcf_gz_file_path

        self.variant_hdf5_file_path = "{}.hdf5".format(self.vcf_gz_file_path)

        self.id_chrom_pickle_gz_file_path = "{}.id_chrom.pickle.gz".format(
            self.vcf_gz_file_path
        )

        self.gene_chrom_pickle_gz_file_path = "{}.gene_chrom.pickle.gz".format(
            self.vcf_gz_file_path
        )

        self.variant_hdf5 = None

        self.id_chrom = {}

        self.gene_chrom = {}

        if not reset:

            try:

                print("Initializing VariantHDF5 ...")

                print("\tReading {} ...".format(self.variant_hdf5_file_path))

                self.variant_hdf5 = open_file(self.variant_hdf5_file_path, mode="r")

                print("\tReading {} ...".format(self.id_chrom_pickle_gz_file_path))

                with gzip_open(
                    self.id_chrom_pickle_gz_file_path
                ) as id_chrom_pickle_gz_file:

                    self.id_chrom = load(id_chrom_pickle_gz_file)

                print("\tReading {} ...".format(self.gene_chrom_pickle_gz_file_path))

                with gzip_open(
                    self.gene_chrom_pickle_gz_file_path
                ) as gene_chrom_pickle_gz_file:

                    self.gene_chrom = load(gene_chrom_pickle_gz_file)

            except (OSError, FileNotFoundError, HDF5ExtError) as exception:

                print("Failed: {}.".format(exception))

                reset = True

        if reset:

            print("Resetting ...")

            if self.variant_hdf5:

                self.variant_hdf5.close()

                print("\tClosed {}.".format(self.variant_hdf5_file_path))

            print("\tMaking {} ...".format(self.variant_hdf5_file_path))

            with gzip_open(self.vcf_gz_file_path) as vcf_gz_file:

                print("Getting data-start position ...")

                data_start_position = None

                line = vcf_gz_file.readline().decode()

                while line.startswith("#"):

                    data_start_position = vcf_gz_file.tell()

                    line = vcf_gz_file.readline().decode()

                print("Counting variants per chrom ...")

                chrom_n_row = defaultdict(lambda: 0)

                n = 0

                chrom = None

                while line:

                    n += 1

                    if not line.startswith("#"):

                        chrom_ = line.split(sep="\t")[0]

                        if chrom != chrom_:

                            print("\t{} ...".format(chrom_))

                            chrom = chrom_

                        chrom_n_row[chrom_] += 1

                    line = vcf_gz_file.readline().decode()

                print("Making {} ...".format(self.variant_hdf5_file_path))

                with open_file(
                    self.variant_hdf5_file_path,
                    mode="w",
                    filters=Filters(complevel=1, complib="blosc"),
                ) as variant_hdf5:

                    chrom_table_row = {}

                    n_per_print = max(1, n // 10)

                    vcf_gz_file.seek(data_start_position)

                    for i, line in enumerate(vcf_gz_file):

                        if i % n_per_print == 0:

                            print("\t{:,}/{:,} ...".format(i + 1, n))

                        line = line.decode(errors="replace").replace("�", "?")

                        if line.startswith("#"):

                            continue

                        chrom, pos, id_, ref, alt, qual, filter_, info, format_, sample = line.split(
                            sep="\t"
                        )

                        if qual == ".":

                            print("Skipped row {} because QUAL==.:\n{}".format(i, line))

                            continue

                        if chrom not in chrom_table_row:

                            print("\t\tMaking {} table ...".format(chrom))

                            chrom_table = variant_hdf5.create_table(
                                "/",
                                "chromosome_{}_variants".format(chrom),
                                description=_VariantDescription,
                                expectedrows=chrom_n_row[chrom],
                            )

                            chrom_table_row[chrom] = chrom_table.row

                        cursor = chrom_table_row[chrom]

                        for id__ in id_.split(sep=";"):

                            cursor["CHROM"] = chrom

                            cursor["POS"] = pos

                            if id__ != ".":

                                cursor["ID"] = id__

                                self.id_chrom[id__] = chrom

                            cursor["REF"] = ref

                            cursor["ALT"] = alt

                            cursor["QUAL"] = qual

                            for info_field in (
                                "CAF",
                                "CLNDISDB",
                                "CLNDN",
                                "CLNSIG",
                                "CLNREVSTAT",
                                "CLNVI",
                            ):

                                info_field_value = get_vcf_info(info, info_field)

                                if info_field_value:

                                    cursor[info_field] = info_field_value

                            for info_ann_field in ("effect", "impact", "gene_name"):

                                info_ann_field_values = get_vcf_info_ann(
                                    info, info_ann_field
                                )

                                if len(info_ann_field_values):

                                    info_ann_field_value_0 = info_ann_field_values[0]

                                    cursor[info_ann_field] = info_ann_field_value_0

                                    if info_ann_field == "gene_name":

                                        self.gene_chrom[info_ann_field_value_0] = chrom

                            cursor["GT"] = get_vcf_sample_format(format_, sample, "GT")

                            cursor.append()

                    print("\tFlushing tables and making column indices ...")

                    for chrom in chrom_table_row:

                        print("\t\t{} table ...".format(chrom))

                        chrom_table = variant_hdf5.get_node(
                            "/", "chromosome_{}_variants".format(chrom)
                        )

                        chrom_table.flush()

                        for column in (
                            "CHROM",
                            "POS",
                            "ID",
                            "REF",
                            "ALT",
                            "QUAL",
                            "CAF",
                            "CLNDISDB",
                            "CLNDN",
                            "CLNSIG",
                            "CLNREVSTAT",
                            "CLNVI",
                            "effect",
                            "impact",
                            "gene_name",
                            "GT",
                        ):

                            chrom_table.cols._f_col(column).create_csindex()

                    self.variant_hdf5 = variant_hdf5

                    print(self.variant_hdf5)

                    print("Writing {} ...".format(self.id_chrom_pickle_gz_file_path))

                    with gzip_open(
                        self.id_chrom_pickle_gz_file_path, mode="wb"
                    ) as id_chrom_pickle_gz_file:

                        dump(self.id_chrom, id_chrom_pickle_gz_file)

                    print("Writing {} ...".format(self.gene_chrom_pickle_gz_file_path))

                    with gzip_open(
                        self.gene_chrom_pickle_gz_file_path, mode="wb"
                    ) as gene_chrom_pickle_gz_file:

                        dump(self.gene_chrom, gene_chrom_pickle_gz_file)

            print("\tReading {} ...".format(self.variant_hdf5_file_path))

            self.variant_hdf5 = open_file(self.variant_hdf5_file_path, mode="r")

    def __del__(self):

        if self.variant_hdf5:

            self.variant_hdf5.close()

            print("Destructor closed {}.".format(self.variant_hdf5_file_path))

    def get_variant_by_id(self, id_):

        chrom = self.id_chrom[id_]

        chrom_table = self.variant_hdf5.get_node(
            "/", "chromosome_{}_variants".format(chrom)
        )

        variant_dicts = read_where_and_map_column_name_on_hdf5_table(
            chrom_table, "ID == b'{}'".format(id_)
        )

        n_variants = len(variant_dicts)

        if n_variants == 1:

            variant_dict = variant_dicts[0]

        else:

            raise ValueError("Found {} variants with ID {}.".format(n_variants, id_))

        make_variant_dict_consistent(variant_dict)

        update_variant_dict(variant_dict)

        return variant_dict

    def get_variants_by_gene(self, gene):

        chrom = self.gene_chrom[gene]

        chrom_table = self.variant_hdf5.get_node(
            "/", "chromosome_{}_variants".format(chrom)
        )

        variant_dicts = read_where_and_map_column_name_on_hdf5_table(
            chrom_table, "gene_name == b'{}'".format(gene)
        )

        for variant_dict in variant_dicts:

            make_variant_dict_consistent(variant_dict)

            update_variant_dict(variant_dict)

        return variant_dicts

    def get_variants_by_region(self, chrom, start_position, end_position):

        chrom_table = self.variant_hdf5.get_node(
            "/", "chromosome_{}_variants".format(chrom)
        )

        variant_dicts = read_where_and_map_column_name_on_hdf5_table(
            chrom_table,
            "({} <= POS) & (POS <= {})".format(start_position, end_position),
        )

        for variant_dict in variant_dicts:

            make_variant_dict_consistent(variant_dict)

            update_variant_dict(variant_dict)

        return variant_dicts
