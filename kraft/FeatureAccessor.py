from collections import defaultdict
from gzip import open as gzip_open
from os import remove
from os.path import isfile
from pickle import dump, load

from tables import Filters, Int32Col, IsDescription, StringCol, open_file

from .read_where_and_map_column_name_on_hdf5_table import (
    read_where_and_map_column_name_on_hdf5_table,
)


class _FeatureHDF5Description(IsDescription):

    seqid = StringCol(256)

    start = Int32Col()

    end = Int32Col()

    Name = StringCol(256)

    biotype = StringCol(256)


class FeatureAccessor:
    def _read_files(self):

        print(f"Reading {self.__class__.__name__} files ...")

        self.feature_hdf5 = open_file(self.feature_hdf5_file_path)

        with gzip_open(self.name_seqid_pickle_gz_file_path) as io:

            self.name_seqid = load(io)

    def _make_files(self):

        print(f"Making {self.__class__.__name__} files ...")

        with gzip_open(
            self.gff3_gz_file_path, mode="rt", encoding="ascii", errors="replace"
        ) as io:

            print("Getting data-start position ...")

            data_start_position = io.tell()

            line = io.readline()

            while line.startswith("#"):

                data_start_position = io.tell()

                line = io.readline()

            print("Counting data ...")

            seqid_n_row = defaultdict(lambda: 0)

            n = 0

            seqid = None

            while line:

                n += 1

                if not line.startswith("#"):

                    seqid_ = line.split(sep="\t")[0]

                    if seqid != seqid_:

                        seqid = seqid_

                        print(seqid)

                    seqid_n_row[seqid] += 1

                line = io.readline()

            print(f"{n:,}")

            if self.feature_hdf5 is not None:

                self.feature_hdf5.close()

                print(f"Closed already opened {self.feature_hdf5_file_path}.")

            print(f"Writing {self.feature_hdf5_file_path} ...")

            with open_file(
                self.feature_hdf5_file_path,
                mode="w",
                filters=Filters(complevel=1, complib="blosc"),
            ) as hdf5:

                seqid_table_row = {}

                name_seqid = {}

                n_per_print = max(1, n // 10)

                io.seek(data_start_position)

                for i, line in enumerate(io):

                    line = line.replace("�", "?")

                    if i % n_per_print == 0:

                        print(f"{i+1:,}/{n:,} ...")

                    if line.startswith("#"):

                        continue

                    seqid, source, type_, start, end, score, strand, phase, attributes = line.split(
                        sep="\t"
                    )

                    if type_ not in self.types:

                        continue

                    if seqid not in seqid_table_row:

                        print(f"Creating table {seqid} ...")

                        seqid_table = hdf5.create_table(
                            "/",
                            f"seqid_{seqid}_features",
                            description=_FeatureHDF5Description,
                            expectedrows=seqid_n_row[seqid],
                        )

                        seqid_table_row[seqid] = seqid_table.row

                    cursor = seqid_table_row[seqid]

                    cursor["seqid"] = seqid

                    cursor["start"] = start

                    cursor["end"] = end

                    name = None

                    biotype = None

                    for attribute in attributes.split(sep=";"):

                        field, value = attribute.split(sep="=")

                        if field == "Name":

                            name = value

                        elif field == "biotype":

                            biotype = value

                    cursor["Name"] = name

                    name_seqid[name] = seqid

                    cursor["biotype"] = biotype

                    cursor.append()

                for seqid in seqid_table_row:

                    print(f"Flushing and making column indices for table {seqid} ...")

                    seqid_table = hdf5.get_node("/", f"seqid_{seqid}_features")

                    seqid_table.flush()

                    for column in ("seqid", "start", "Name"):

                        seqid_table.cols._f_col(column).create_csindex()

                print(hdf5)

        print(f"Writing {self.name_seqid_pickle_gz_file_path} ...")

        with gzip_open(self.name_seqid_pickle_gz_file_path, mode="wb") as io:

            dump(name_seqid, io)

    def __init__(self, gff3_gz_file_path, types=("gene",), reset=False):

        self.gff3_gz_file_path = gff3_gz_file_path

        self.feature_hdf5_file_path = f"{self.gff3_gz_file_path}.hdf5"

        self.name_seqid_pickle_gz_file_path = (
            f"{self.gff3_gz_file_path}.name_seqid.pickle.gz"
        )

        self.types = types

        self.feature_hdf5 = None

        self.name_seqid = {}

        try:

            self._read_files()

        except Exception as exception:

            print(exception)

            reset = True

        if reset:

            print(f"Resettting {self.__class__.__name__} ...")

            for file_path in (
                self.feature_hdf5_file_path,
                self.name_seqid_pickle_gz_file_path,
            ):

                if isfile(file_path):

                    remove(file_path)

            self._make_files()

            self._read_files()

    def __del__(self):

        if self.feature_hdf5 is not None:

            self.feature_hdf5.close()

            print(f"Destructor closed {self.feature_hdf5_file_path}.")

    def get_features_by_name(self, name):

        seqid = self.name_seqid[name]

        seqid_table = self.feature_hdf5.get_node("/", f"seqid_{seqid}_features")

        feature_dicts = read_where_and_map_column_name_on_hdf5_table(
            seqid_table, f"Name == b'{name}'"
        )

        return feature_dicts

    def get_features_by_region(self, seqid, start_position, end_position):

        seqid_table = self.feature_hdf5.get_node("/", f"seqid_{seqid}_features")

        feature_dicts = read_where_and_map_column_name_on_hdf5_table(
            seqid_table, f"({start_position} <= start) & (start <= {end_position})"
        )

        return feature_dicts
