from pandas import read_csv
from tables import NoSuchNodeError

from .count_gene_impacts_from_variant_dicts import count_gene_impacts_from_variant_dicts
from .FeatureHDF5 import FeatureHDF5
from .VariantHDF5 import VariantHDF5


class Genome:
    def __init__(
        self,
        reference_fasta_gz_file_path,
        reference_gff3_gz_file_path,
        vcf_gz_file_path,
        reset=False,
    ):

        self.reference_fasta_gz_file_path = reference_fasta_gz_file_path

        self.chromosome_size = read_csv(
            "{}.fai".format(self.reference_fasta_gz_file_path),
            sep="\t",
            header=None,
            usecols=(0, 1),
            index_col=0,
            squeeze=True,
        ).to_dict()

        self.reference_gff3_gz_file_path = reference_gff3_gz_file_path

        self.reference_gene_hdf5 = FeatureHDF5(
            self.reference_gff3_gz_file_path, reset=reset
        )

        self.vcf_gz_file_path = vcf_gz_file_path

        self.variant_hdf5 = VariantHDF5(self.vcf_gz_file_path, reset=reset)

    def explore_genome_by_variant(self, variant_id):

        try:

            variant_dicts = [self.variant_hdf5.get_variant_by_id(variant_id)]

        except KeyError as exception:

            print("KeyError: {}.".format(exception))

            variant_dicts = []

        return {"gene_dicts": [], "variant_dicts": variant_dicts}

    def explore_genome_by_gene(self, gene):

        try:

            gene_dicts = self.reference_gene_hdf5.get_features_by_name(gene)

        except KeyError as exception:

            print("KeyError: {}.".format(exception))

            gene_dicts = []

        if 1 < len(gene_dicts):

            print("{} matches multiple genes, but using only the 1st one.".format(gene))

            gene_dicts = gene_dicts[:1]

        try:

            variant_dicts = self.variant_hdf5.get_variants_by_gene(gene)

            looked_for_variants = True

        except KeyError as exception:

            print("KeyError: {}.".format(exception))

            if len(gene_dicts) == 1:

                print(
                    "VariantHDF5 does not have gene {}, so using gene region (from FeatureHDF5) to get variants ...".format(
                        gene
                    )
                )

                try:

                    variant_dicts = self.variant_hdf5.get_variants_by_region(
                        gene_dicts[0]["seqid"],
                        int(gene_dicts[0]["start"]),
                        int(gene_dicts[0]["end"]),
                    )

                    looked_for_variants = True

                except NoSuchNodeError as exception:

                    print("NoSuchNodeError: {}.".format(exception))

                    variant_dicts = []

                    looked_for_variants = False

            else:

                variant_dicts = []

                looked_for_variants = False

        for gene_dict in gene_dicts:

            if looked_for_variants:

                impacts = count_gene_impacts_from_variant_dicts(
                    variant_dicts, gene_dict["Name"]
                )

            else:

                impacts = {"HIGH": "?", "MODERATE": "?", "LOW": "?", "MODIFIER": "?"}

            gene_dict["impacts"] = impacts

        return {"gene_dicts": gene_dicts, "variant_dicts": variant_dicts}

    def explore_genome_by_region(self, chromosome, start_position, end_position):

        try:

            gene_dicts = self.reference_gene_hdf5.get_features_by_region(
                chromosome, start_position, end_position
            )

        except NoSuchNodeError as exception:

            print("NoSuchNodeError: {}.".format(exception))

            gene_dicts = []

        try:

            variant_dicts = self.variant_hdf5.get_variants_by_region(
                chromosome, start_position, end_position
            )

            looked_for_variants = True

        except NoSuchNodeError as exception:

            print("NoSuchNodeError: {}.".format(exception))

            variant_dicts = []

            looked_for_variants = False

        for gene_dict in gene_dicts:

            if looked_for_variants:

                impacts = count_gene_impacts_from_variant_dicts(
                    variant_dicts, gene_dict["Name"]
                )

            else:

                impacts = {"HIGH": "?", "MODERATE": "?", "LOW": "?", "MODIFIER": "?"}

            gene_dict["impacts"] = impacts

        return {"gene_dicts": gene_dicts, "variant_dicts": variant_dicts}
