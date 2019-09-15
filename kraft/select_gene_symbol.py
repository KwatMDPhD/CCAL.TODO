from os.path import join

from numpy import asarray
from pandas import isna, read_csv

from .DATA_DIRECTORY_PATH import DATA_DIRECTORY_PATH


def select_gene_symbol(
    gene_family_name_to_remove=(
        "18S ribosomal RNAs",
        "28S ribosomal RNAs",
        "45S pre-ribosomal RNAs",
        "5.8S ribosomal RNAs",
        "5S ribosomal RNAs",
        "Cytoplasmic transfer RNAs",
        "Long non-coding RNAs (published)",
        "MicroRNAs",
        "Mitochondrially encoded ribosomal RNAs",
        "Mitochondrially encoded tRNAs",
        "Nuclear-encoded mitochondrial transfer RNAs",
        "Piwi-interacting RNA clusters",
        "Ribosomal 45S RNA clusters",
        "Ribosomal 45S rRNA genes outside of clusters",
        "RNAs, 7SL, cytoplasmic",
        "RNAs, Ro-associated Y",
        "Small Cajal body-specific RNAs",
        "Small NF90 (ILF3) associated RNAs",
        "Small nuclear RNAs",
        "Small nucleolar RNAs, C/D box",
        "Small nucleolar RNAs, H/ACA box",
        "Vault RNAs",
        "L ribosomal proteins",
        "Mitochondrial ribosomal proteins",
        "S ribosomal proteins",
        "Mitochondrial complex II: succinate dehydrogenase subunits",
        "Mitochondrial complex III: ubiquinol-cytochrome c reductase complex subunits",
        "Mitochondrial complex IV: cytochrome c oxidase subunits",
        "Mitochondrial complex V: ATP synthase subunits",
        "NADH:ubiquinone oxidoreductase core subunits",
        "NADH:ubiquinone oxidoreductase supernumerary subunits",
    ),
    locus_type_to_keep=("gene with protein product",),
):

    hgnc = read_csv(join(DATA_DIRECTORY_PATH, "hgnc.tsv"), sep="\t", index_col=0)

    remove_by_gene_family_name = asarray(
        tuple(
            not isna(gene_family_name)
            and any(str_ in gene_family_name for str_ in gene_family_name_to_remove)
            for gene_family_name in hgnc["Gene Family Name"]
        )
    )

    print(
        "Removing {}/{} based on Gene Family Name...".format(
            remove_by_gene_family_name.sum(), remove_by_gene_family_name.size
        )
    )

    keep_by_locus_type = asarray(
        tuple(
            not isna(locus_type)
            and any(str_ in locus_type for str_ in locus_type_to_keep)
            for locus_type in hgnc["Locus Type"]
        )
    )

    print(
        "Keeping {}/{} based on Locus Type...".format(
            keep_by_locus_type.sum(), keep_by_locus_type.size
        )
    )

    is_removed = remove_by_gene_family_name | ~keep_by_locus_type

    for column_name in ("Gene Family Name", "Locus Type"):

        dataframe = hgnc.loc[is_removed, column_name].value_counts().to_frame()

        dataframe.index.name = column_name

        dataframe.columns = ("N Removed",)

        print(dataframe)

    gene_symbols = (
        hgnc.loc[~is_removed, ["Approved Symbol", "Previous Symbols"]]
        .unstack()
        .dropna()
        .unique()
    )

    print("Selected {} gene symbols.".format(gene_symbols.size))

    return gene_symbols.tolist()
