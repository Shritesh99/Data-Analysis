
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
train = pd.read_csv('all/train.csv')
test = pd.read_csv('all/test.csv')
test_result = pd.read_csv('all/gender_submission.csv')

# Selecting desireable fields
x_train = train.iloc[:, [2, 4, 5, 6, 7, 9, 10, 11]].values
y_train = train.iloc[:, 1].values

x_test = test.iloc[:, [1, 3, 4, 5, 6, 8, 9, 10]].values
y_test = test_result.iloc[:, 1].values

# Data Preprocessing

# Taking care of missing data
