#######################################
###### Decision Tree in Python ########
#######################################

import math
import pprint

pp = pprint.PrettyPrinter(indent=4)

def entropy(p, n):
    """
    Calculates the entropy for one attribute using the number of occurrences
    """
    # This avoids some log(0) issues
    if p == 0 or n == 0:
        return 0

    # Convert from a number of occurrences to a probability 
    p1 = p / (p + n)
    n1 = n / (p + n)

    return -1 * p1 * math.log(p1, 2) - n1 * math.log(n1, 2)

def entropy2(counts):
    """
    Calculates the entropy for two attributes given a list of [p, n]
    """

    # Calculate the total
    total = 0
    for row in counts:
        total += row[0] + row[1]

    # Calculate the entropy for the two attributes
    entropy2 = 0
    for row in counts:
        p = row[0]
        n = row[1]
        occurrences = p + n
        entropy2 += occurrences / total * entropy(p, n)
    return entropy2

def select_attribute(data):
    """
    Chooses the attribute, by index, that would net the greatest information
    gain. Assumes the last entry in a row is the boolean class
    """

    # Calculate the total entropy
    p = 0
    n = 0
    for row in data:
        if row[-1]:
            p += 1
        else:
            n += 1
    total_entropy = entropy(p, n)

    # Calculate the gain for each attribute
    for index in range(len(data[0]) - 1):
        counts = {}
        for row in data:
            attribute_value = row[index]
            if attribute_value not in counts:
                counts[attribute_value] = [0, 0]
            if row[-1]:
                counts[attribute_value][0] += 1
            else:
                counts[attribute_value][1] += 1
        gain = total_entropy - entropy2(counts.values())
        print(f"Index: {index} Gain: {gain}")

select_attribute([
    ['<75',     '1', True],
    ['75..150', '2', True],
    ['<75',     '1', False],
    ['>150',    '2', True],
    ['75..150', '1', False],
    ['75..150', '2', True],
    ['<75',     '2', False],
    ['>150',    '2', True],
    ['<75',     '1', False],
    ['75..150', '1', True],
    ['75..150', '2', True],
    ['>150',    '1', True],
    ['<75',     '1', False],
    ['<75',     '2', False],
    ['75..150', '2', True],
])
