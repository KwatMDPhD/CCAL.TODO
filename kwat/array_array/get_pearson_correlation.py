from scipy.stats import pearsonr


def get_pearson_correlation(ve0, ve1):

    return pearsonr(ve0, ve1)[0]
