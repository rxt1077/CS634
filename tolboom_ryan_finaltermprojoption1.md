# Final Term Project: Option 1 - Supervised Data Mining (Classification)

Classification algorithms used:

* Category 3 (Decision Trees) -
* Category 5 (Naive Bayes) - NaiveBayes

Dataset: [Breast Cancer Wisconsin (Diagnostic) Data Set](http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)

## Preparation of Dataset

The dataset includes 683 complete instances. In preparation for the 10-fold cross
validation method it was broken into ten parts and instances with unknown
attribute values were removed. Concatenations of the nine remaining parts were
created to use as training data. The commands used are shown below:

```bash
$ grep -v "?" breast-cancer-wisconsin.data | wc -l
683
$ grep -v "?" breast-cancer-wisconsin.data | \
> split -d -l 69 - breast-cancer-wisconsin-part-
$ for FILE in *part*; GLOBIGNORE="$FILE:*.train"; cat *part* > $FILE.train; done
$ ls -sh1 *part*
4.0K breast-cancer-wisconsin-part-00
 32K breast-cancer-wisconsin-part-00.train
4.0K breast-cancer-wisconsin-part-01
 32K breast-cancer-wisconsin-part-01.train
4.0K breast-cancer-wisconsin-part-02
 32K breast-cancer-wisconsin-part-02.train
4.0K breast-cancer-wisconsin-part-03
 32K breast-cancer-wisconsin-part-03.train
4.0K breast-cancer-wisconsin-part-04
 20K breast-cancer-wisconsin-part-04.train
4.0K breast-cancer-wisconsin-part-05
 20K breast-cancer-wisconsin-part-05.train
4.0K breast-cancer-wisconsin-part-06
 32K breast-cancer-wisconsin-part-06.train
4.0K breast-cancer-wisconsin-part-07
 20K breast-cancer-wisconsin-part-07.train
4.0K breast-cancer-wisconsin-part-08
 20K breast-cancer-wisconsin-part-08.train
4.0K breast-cancer-wisconsin-part-09
 20K breast-cancer-wisconsin-part-09.train
```

## Naive Bayes

The naive bayes algorithm was developed in Python. The first argument is a set
of training data to use and the second argument is the test data to predict
labels for. The output is a count of True Positives, False Positives, True
Negatives, and False Negatives.

### Source Code

```python
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
```

### Output

The program was run in a loop on all ten parts created earlier. The command to
run the program as well as the program's output can be see below.

```bash
$ for PART in 00 01 02 03 04 05 06 07 08 09
> do
> python3 ./naive_bayes.py \
> breast-cancer-wisconsin/breast-cancer-wisconsin-part-$PART.train \
> breast-cancer-wisconsin/breast-cancer-wisconsin-part-$PART
> done
{'TP': 34, 'FP': 5, 'TN': 30, 'FN': 0}
{'TP': 27, 'FP': 6, 'TN': 36, 'FN': 0}
{'TP': 28, 'FP': 3, 'TN': 38, 'FN': 0}
{'TP': 38, 'FP': 5, 'TN': 25, 'FN': 1}
{'TP': 33, 'FP': 4, 'TN': 32, 'FN': 0}
{'TP': 15, 'FP': 5, 'TN': 49, 'FN': 0}
{'TP': 19, 'FP': 5, 'TN': 45, 'FN': 0}
{'TP': 10, 'FP': 2, 'TN': 57, 'FN': 0}
{'TP': 23, 'FP': 2, 'TN': 44, 'FN': 0}
{'TP': 11, 'FP': 2, 'TN': 49, 'FN': 0}
```

### Average Results over 10-Fold Cross Validation

| True Positive | True Negative | False Positive | False Negative |
| ------------- | ------------- | -------------- | -------------- |
| 23.8          | 40.5          | 3.9            | 0.1            |

As can be seen, this method classifies data with a 94% accuracy.
