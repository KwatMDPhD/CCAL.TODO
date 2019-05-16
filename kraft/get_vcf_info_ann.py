from .get_vcf_info import get_vcf_info
from .VCF_ANN_KEYS import VCF_ANN_KEYS


def get_vcf_info_ann(info, key, n_ann=None):

    ann = get_vcf_info(info, "ANN")

    if ann is not None:

        i = VCF_ANN_KEYS.index(key)

        return tuple(ann_.split(sep="|")[i] for ann_ in ann.split(sep=",")[:n_ann])
