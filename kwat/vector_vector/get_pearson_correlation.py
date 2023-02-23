from scipy.stats import pearsonr


def get_pearson_correlation(ve1, ve2):
    return pearsonr(ve1, ve2)[0]
