import numpy as np
from scipy.stats import norm

def get_sigma(alpha):
        return 0.5 / norm.ppf(alpha)

def get_alpha(sigma):
    return norm.cdf(0.5, scale=sigma)
