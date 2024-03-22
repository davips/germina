from scipy.stats import binom


def p_value(accuracy, num_samples):
    hits = int(accuracy * num_samples)
    p_value = 1 - binom.cdf(hits - 1, num_samples, 0.5)
    return p_value
