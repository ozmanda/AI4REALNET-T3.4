import numpy as np
from numbers import Number
from typing import Dict

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


def merge_dicts(source_dict: Dict, destination_dict: Dict) -> Dict:
    """ 
    Merge dictionaries that potentially contain the same keys. The following cases are considered:
        1. The key is not in the destination dictionary, in which case the key-value pair is added to the destination dictionary.
        2. The key is in the destination dictionary. If the value is a:
              2.1 number, the values are summed
              2.2 ndarray, the arrays are summed
              2.3 list, the destination list is extended
              2.4 two non-number values are turned into a list
     
    Input: 
        - source_dict       the dictionary from which data is transferred       dict
        - destination_dict  the dictionary to which data is transferred         dict

    Return:
        - destination_dict  the dictionary with the merged data                 dict
            
    """
    for key, value in source_dict.items():
        if not key in destination_dict: 
            destination_dict[key] = value
        
        elif isinstance(value, Number):
            destination_dict[key] = destination_dict.get(key, 0) + value
        
        elif isinstance(value, np.ndarray): # in the case of multi-agent rewards
            destination_dict[key] += value

        else: # covers the various list cases
            if isinstance(value, list) and isinstance(destination_dict[key], list): # 2.3 mmerges two lists
                destination_dict[key].extend(value)
            elif isinstance(destination_dict[key], list): # 2.3 one list, one number
                destination_dict[key].append(value)
            else: # 2.4 two non-number values 
                destination_dict[key] = [destination_dict[key], value]
    return destination_dict
