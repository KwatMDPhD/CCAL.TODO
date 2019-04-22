"""
Reads a GCT into a Pandas dataframe
"""
from pandas import DataFrame, read_csv

def A_gsea_load_gct(filepath):
    gct = read_csv(filepath, sep='\t', index_col=0, skiprows=2)
    gct = gct.drop(["DESCRIPTION"], axis=1)
    return gct
