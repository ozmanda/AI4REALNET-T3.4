import numpy as np

def max_lowerthan(sequence, value):
    """ Find the largest item in sequence that is lower than a given value. """
    max = -np.inf
    idx = len(sequence) - 1
    while idx >= 0:
        if sequence[idx] <= value and sequence[idx] > max:
            max = sequence[idx]
        idx -= 1
    return max

def min_greaterthan(sequence, value):
    """ Find the smallest item in sequence that is greater than a given value. """
    min = np.inf
    idx = len(sequence) - 1
    while idx >= 0:
        if sequence[idx] >= value and sequence[idx] < min:
            min = sequence[idx]
        idx -= 1
    return min