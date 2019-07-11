#######################################
###### Decision Tree in Python ########
#######################################
# Usage is python3 decision_tree.py <training data> <test data>

import math
import pprint
import csv
import sys

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

    # Calculate the gain for each attribute and keep track of the index of the
    # max
    max_index = 0
    max_gain = 0
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
        #print(f"Index: {index} Gain: {gain}")
        if gain > max_gain:
            max_gain = gain
            max_index = index

    return max_index

def split_on_attribute(data, index):
    """
    Takes the current data and breaks it into groups based by an attribute index
    This attribute is removed from the resulting groups
    """
    groups = {}
    for row in data:
        attribute_value = row[index]
        # Since we are already making a decision based on this attribute we don't
        # need to include it in the lower nodes
        row.pop(index)
        if attribute_value not in groups:
            groups[attribute_value] = []
        groups[attribute_value].append(row)
    return groups

def majority_class(data):
    """
    Counts the classes of the rows in a dataset to determine if they are
    unanimous and what the majority is
    """
    p = 0
    n = 0
    for row in data:
        if row[-1]:
            p += 1
        else:
            n += 1
    result = {}
    result['unanimous'] = True if (p == 0 or n == 0) else False
    result['majority'] = p > n
    return result

class Node:
    """
    A general class for the nodes of a tree
    """
    type = None
    index = None
    majority = None
    branches = None

def build_tree(data):
    """
    Builds a decision tree from a dataset
    """
    #print("Creating node from data...")
    #pp.pprint(data)
    node = Node()

    # Check to see if all the labels are the same, if so we are creating a RESULT
    # node
    result = majority_class(data)
    node.majority = result['majority']
    if result['unanimous']:
        #print(f"RESULT: {result['majority']}")
        node.type = 'RESULT'
        return node

    # If not we are creating a DECISION node
    node.type = 'DECISION'
    index = select_attribute(data)
    node.index = index
    node.branches = {}
    #print(f"DECISION: Splitting on index {index}...")
    groups = split_on_attribute(data, index)
    for attribute_value, group_data in groups.items():
        #print(f"Creating {attribute_value} node")
        node.branches[attribute_value] = build_tree(group_data)
    return node

# Testing data for tree building
#
#data = [
#    ['<75',     '1', 'City', 'MCI',    True],
#    ['75..150', '2', 'Town', 'MCI',    True],
#    ['<75',     '1', 'City', 'Sprint', False],
#    ['>150',    '2', 'Town', 'AT&T',   True],
#    ['75..150', '1', 'City', 'MCI',    False],
#    ['75..150', '2', 'Town', 'AT&T',   True],
#    ['<75',     '2', 'Town', 'AT&T',   False],
#    ['>150',    '2', 'City', 'Sprint', True],
#    ['<75',     '1', 'City', 'AT&T',   False],
#    ['75..150', '1', 'Town', 'MCI',    True],
#    ['75..150', '2', 'City', 'Sprint', True],
#    ['>150',    '1', 'Town', 'AT&T',   True],
#    ['<75',     '1', 'Town', 'AT&T',   False],
#    ['<75',     '2', 'City', 'Sprint', False],
#    ['75..150', '2', 'City', 'MCI',    True],
#]
#root = build_tree(data)
#import pdb; pdb.set_trace()

def classify_row(row, node):
    """
    Uses a decision tree to classify a row
    """
    if node.type == 'RESULT':
        return node.majority
    else:
        attribute_value = row[node.index]
        if attribute_value not in node.branches:
            #print("Unable to find unanimous result, using majority")
            return node.majority
        else:
            return classify_row(row, node.branches[attribute_value])

# Load the training data from the file
data = []
with open(sys.argv[1], newline='') as csvfile:
    trainingdata = csv.reader(csvfile)
    for row in trainingdata:
        # Our training data uses 2 for benign and '4' for malignant
        if row[10] == '4':
            row[10] = True
        else:
            row[10] = False
        # We don't need the first column, it is an ID
        data.append(row[1:])

# Recursively create the decision tree from the data
root = build_tree(data)

# Load the test data and classify each row keep track of TP, TN, FP, FN
results = { 'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0 }
with open(sys.argv[2], newline='') as csvfile:
    testdata = csv.reader(csvfile)
    for row in testdata:
        if classify_row(row[1:9], root): # Positive
            if row[10] == '4': # True positive
                results['TP'] += 1
            else: # False positive
                results['FP'] += 1
        else: # Negative
            if row[10] == '2': # True negative
                results['TN'] += 1
            else: # False negative
                results['FN'] += 1

print(results)
