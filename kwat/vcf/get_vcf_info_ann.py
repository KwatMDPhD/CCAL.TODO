from .ANN_KEYS import ANN_KEYS
from .get_vcf_info import get_vcf_info


def get_vcf_info_ann(io, ke, n_an=None):

    an = get_vcf_info(io, "ANN")

    if an is not None:

        ie = ANN_KEYS.index(ke)

        return [ans.split(sep="|")[ie] for ans in an.split(sep=",")[:n_an]]
