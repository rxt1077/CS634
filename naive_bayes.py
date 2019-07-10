####################################################
########## Naive Bayes in Python ###################
####################################################
# Usage is: python3 naive_bayes.py <training_data> <test_data>

import sys
import math
import csv
import pprint
pp = pprint.PrettyPrinter(indent=4)

# Training

# Initialize the data
means = {
    'benign'   : [0,0,0,0,0,0,0,0,0],
    'malignant': [0,0,0,0,0,0,0,0,0],
}
counts = {
    'benign'   : 0,
    'malignant': 0,
}
stddevs = {
    'benign'   : [0,0,0,0,0,0,0,0,0],
    'malignant': [0,0,0,0,0,0,0,0,0],
}

# Read from the training file
with open(sys.argv[1], newline='') as csvfile:
    trainingdata = csv.reader(csvfile)

    # Sum up the columns for each class and count instances
    for row in trainingdata:
        # There are two classes in this dataset 2: benign and 4: malignant
        if row[10] == '2':
            label = 'benign'
        else:
            label = 'malignant'
        counts[label] += 1
        # The first column is an ID which we don't need for training
        for col in range(1, 10):
            means[label][col - 1] += int(row[col])

    # Divide by the count to get the mean
    for label in ['benign', 'malignant']:
        for col in range(9):
            means[label][col] = means[label][col] / counts[label]

    # Calculate the standard deviation
    # Iterate through the dataset again
    csvfile.seek(0)
    for row in trainingdata:
        # There are two classes in this dataset 2: benign and 4: malignant
        if row[10] == '2':
            label = 'benign'
        else:
            label = 'malignant'
        # The first column is an ID which we don't need for training
        # sum the squares of the distance from the mean
        for col in range(1, 10):
            stddevs[label][col - 1] += (int(row[col]) - means[label][col - 1])**2
    # divide by the count to get the average and take the square root
    for label in ['benign', 'malignant']:
        for col in range(9):
            stddevs[label][col] = math.sqrt(stddevs[label][col] / counts[label])

# Prediction

results = {
    'TP': 0,
    'FP': 0,
    'TN': 0,
    'FN': 0,
}

with open(sys.argv[2], newline='') as csvfile:
    testdata = csv.reader(csvfile)
    for row in testdata:
        distances = {
            'benign':    0,
            'malignant': 0,
        }
        for col in range(1, 10):
            for label in ['benign', 'malignant']:
                # Calculate the squared distance normalized by stddev for each
                distances[label] += ((means[label][col - 1] - int(row[col])) /
                                     stddevs[label][col - 1])**2
        if distances['malignant'] < distances['benign']:
            # A positive prediction
            if row[10] == '4':
                # True positive
                results['TP'] += 1
            else:
                # False positive
                results['FP'] += 1
        else:
            # A negative prediction
            if row[10] == '2':
                # True negative
                results['TN'] += 1
            else:
                # False negative
                results['FN'] += 1

print(results)
