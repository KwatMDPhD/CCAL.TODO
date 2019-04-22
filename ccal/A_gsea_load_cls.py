"""
Reads cls into list of class IDs
"""

def A_gsea_load_cls(filepath):
    cls_symbols = open(filepath).readlines()[2].split()
    cls_unique_map = {symbol: i for i, symbol in enumerate(set(cls_symbols))}
    cls_ints = [cls_unique_map[symbol] for symbol in cls_symbols]
    return cls_ints
