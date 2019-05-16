from collections import defaultdict
from gzip import open as gzip_open
from os import remove
from os.path import isfile
from pickle import dump, load

from tables import Filters, Float32Col, Int32Col, IsDescription, StringCol, open_file

from .get_vcf_info import get_vcf_info
from .get_vcf_info_ann import get_vcf_info_ann
from .get_vcf_sample_format import get_vcf_sample_format
from .make_variant_dict_consistent import make_variant_dict_consistent
from .read_where_and_map_column_name_on_hdf5_table import (
    read_where_and_map_column_name_on_hdf5_table,
)
from .update_variant_dict import update_variant_dict


class _VariantHDF5Description(IsDescription):

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


class VariantAccessor:
    def _read_files(self):

        print(f"Reading {self.__class__.__name__} files ...")

        self.variant_hdf5 = open_file(self.variant_hdf5_file_path, mode="r")

        with gzip_open(self.id_chrom_pickle_gz_file_path) as io:

            self.id_chrom = load(io)

        with gzip_open(self.gene_chrom_pickle_gz_file_path) as io:

            self.gene_chrom = load(io)

    def _make_files(self):

        print(f"Making {self.__class__.__name__} files ...")

        with gzip_open(
            self.vcf_gz_file_path, mode="rt", encoding="ascii", errors="replace"
        ) as io:

            print("Getting data-start position ...")

            data_start_position = io.tell()

            line = io.readline()

            while line.startswith("#"):

                data_start_position = io.tell()

                line = io.readline()

            print("Counting data ...")

            chrom_n = defaultdict(lambda: 0)

            n = 0

            chrom = None

            while line:

                n += 1

                if not line.startswith("#"):

                    chrom_ = line.split(sep="\t")[0]

                    if chrom != chrom_:

                        chrom = chrom_

                        print(chrom)

                    chrom_n[chrom] += 1

                line = io.readline()

            print(f"{n:,}")

            if self.variant_hdf5 is not None:

                self.variant_hdf5.close()

                print(f"Closed already opened {self.variant_hdf5_file_path}.")

            print(f"Writing {self.variant_hdf5_file_path} ...")

            with open_file(
                self.variant_hdf5_file_path,
                mode="w",
                filters=Filters(complevel=1, complib="blosc"),
            ) as hdf5:

                chrom_table_row = {}

                id_chrom = {}

                gene_chrom = {}

                n_per_print = max(1, n // 10)

                io.seek(data_start_position)

                for i, line in enumerate(io):

                    line = line.replace("�", "?")

                    if i % n_per_print == 0:

                        print(f"{i+1:,}/{n:,} ...")

                    if line.startswith("#"):

                        continue

                    chrom, pos, id_, ref, alt, qual, filter_, info, format_, sample = line.split(
                        sep="\t"
                    )

                    if chrom not in chrom_table_row:

                        print(f"Creating table {chrom} ...")

                        chrom_table = hdf5.create_table(
                            "/",
                            f"chromosome_{chrom}_variants",
                            description=_VariantHDF5Description,
                            expectedrows=chrom_n[chrom],
                        )

                        chrom_table_row[chrom] = chrom_table.row

                    cursor = chrom_table_row[chrom]

                    for id__ in id_.split(sep=";"):

                        cursor["CHROM"] = chrom

                        cursor["POS"] = pos

                        if id__ != ".":

                            cursor["ID"] = id__

                            id_chrom[id__] = chrom

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

                            if info_field_value is not None:

                                cursor[info_field] = info_field_value

                        for info_ann_field in ("effect", "impact", "gene_name"):

                            info_ann_field_values = get_vcf_info_ann(
                                info, info_ann_field
                            )

                            if 0 < len(info_ann_field_values):

                                info_ann_field_value_0 = info_ann_field_values[0]

                                cursor[info_ann_field] = info_ann_field_value_0

                                if info_ann_field == "gene_name":

                                    gene_chrom[info_ann_field_value_0] = chrom

                        cursor["GT"] = get_vcf_sample_format(format_, "GT", sample)

                        cursor.append()

                for chrom in chrom_table_row:

                    print(f"Flushing and making column indices for table {chrom} ...")

                    chrom_table = hdf5.get_node("/", f"chromosome_{chrom}_variants")

                    chrom_table.flush()

                    for column in ("CHROM", "POS", "ID", "gene_name"):

                        chrom_table.cols._f_col(column).create_csindex()

                print(hdf5)

        print(f"Writing {self.id_chrom_pickle_gz_file_path} ...")

        with gzip_open(self.id_chrom_pickle_gz_file_path, mode="wb") as io:

            dump(id_chrom, io)

        print(f"Writing {self.gene_chrom_pickle_gz_file_path} ...")

        with gzip_open(self.gene_chrom_pickle_gz_file_path, mode="wb") as io:

            dump(gene_chrom, io)

    def __init__(self, vcf_gz_file_path, reset=False):

        self.vcf_gz_file_path = vcf_gz_file_path

        self.variant_hdf5_file_path = f"{self.vcf_gz_file_path}.hdf5"

        self.id_chrom_pickle_gz_file_path = (
            f"{self.vcf_gz_file_path}.id_chrom.pickle.gz"
        )

        self.gene_chrom_pickle_gz_file_path = (
            f"{self.vcf_gz_file_path}.gene_chrom.pickle.gz"
        )

        self.variant_hdf5 = None

        self.id_chrom = {}

        self.gene_chrom = {}

        try:

            self._read_files()

        except Exception as exception:

            print(exception)

            reset = True

        if reset:

            print(f"Resettting {self.__class__.__name__} ...")

            for file_path in (
                self.variant_hdf5_file_path,
                self.id_chrom_pickle_gz_file_path,
                self.gene_chrom_pickle_gz_file_path,
            ):

                if isfile(file_path):

                    remove(file_path)

            self._make_files()

            self._read_files()

    def __del__(self):

        if self.variant_hdf5 is not None:

            self.variant_hdf5.close()

            print(f"Destructor closed {self.variant_hdf5_file_path}.")

    def get_variant_by_id(self, id_):

        chrom = self.id_chrom[id_]

        chrom_table = self.variant_hdf5.get_node("/", f"chromosome_{chrom}_variants")

        variant_dicts = read_where_and_map_column_name_on_hdf5_table(
            chrom_table, f"ID == b'{id_}'"
        )

        n_variants = len(variant_dicts)

        if n_variants == 1:

            variant_dict = variant_dicts[0]

        else:

            raise ValueError(f"Found {n_variants} variant with ID {id_}.")

        make_variant_dict_consistent(variant_dict)

        update_variant_dict(variant_dict)

        return variant_dict

    def get_variants_by_gene(self, gene):

        chrom = self.gene_chrom[gene]

        chrom_table = self.variant_hdf5.get_node("/", f"chromosome_{chrom}_variants")

        variant_dicts = read_where_and_map_column_name_on_hdf5_table(
            chrom_table, f"gene_name == b'{gene}'"
        )

        for variant_dict in variant_dicts:

            make_variant_dict_consistent(variant_dict)

            update_variant_dict(variant_dict)

        return variant_dicts

    def get_variants_by_region(self, chrom, start_position, end_position):

        chrom_table = self.variant_hdf5.get_node("/", f"chromosome_{chrom}_variants")

        variant_dicts = read_where_and_map_column_name_on_hdf5_table(
            chrom_table, f"({start_position} <= POS) & (POS <= {end_position})"
        )

        for variant_dict in variant_dicts:

            make_variant_dict_consistent(variant_dict)

            update_variant_dict(variant_dict)

        return variant_dicts
