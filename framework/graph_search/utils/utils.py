import numpy as np


def calc_relative_error(ground_true_value: float, erroneous_value: float):
    return abs(ground_true_value - erroneous_value) / ground_true_value if ground_true_value > 0 else np.inf
