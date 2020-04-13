# -*- coding: utf-8 -*-

''' Mini project for CMPUT 466

Creation Date: April 11, 2020
Author: Luke Slevinsky

This project explores various classification algorithms on one of the
most popular multi class classification datasets, the iris dataset,
which can be found at: http://archive.ics.uci.edu/ml/datasets/Iris
'''

import os
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.special import expit
# sklearn
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

# Data viz
import seaborn as sns
import matplotlib.pyplot as plt


def read_iris_data():
    # Reading data from CSV file
    names = ['sepal_length', 'sepa_width',
             'petal_length', 'petal_width', 'class']
    parent_dir_path = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(
        parent_dir_path, 'bezdekIris.data'), names=names, engine='python')

    # scatter plot matrix
    # pd.plotting.scatter_matrix(df)
    # plt.show()

    # Check for any null values
    missing_df = df.isnull().sum()
    print("Nulls per column")
    print(missing_df)
    print()

    X = df.iloc[:, :4]
    t = df.loc[:, 'class']

    return X, t


def stratKFold(X, t, classifier):
    accuracies = []
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=403)
    skf.get_n_splits(X, t)

    for train_index, test_index in skf.split(X, t):
        print(f'Train:{train_index} Validation:{test_index}')
        X1_train, X1_test = X.iloc[train_index], X.iloc[test_index]
        t1_train, t1_test = t.iloc[train_index], t.iloc[test_index]

        classifier.fit(X1_train, t1_train)
        prediction = classifier.predict(X1_test)
        score = accuracy_score(prediction, t1_test)
        accuracies.append(score)
    return np.array(accuracies).mean()


def baseline_prediction(X_train, X_test, t_train, t_test, strategy='most_frequent'):
    classifier = DummyClassifier(strategy=strategy)
    classifier.fit(X_train, t_train)
    val_acc = classifier.score(X_train, t_train)

    use_model(classifier, val_acc, X_test, t_test, 'Majority Guess')


def logistic_regression_prediction(X_train, X_test, t_train, t_test):
    classifier = LogisticRegression(random_state=random_seed, max_iter=10000)
    k_fold = KFold(n_splits=10, shuffle=True, random_state=random_seed)

    grid = {
        # , 'penalty': ['l1', 'l2']
        'C': np.power(10.0, np.arange(-10, 10))
    }
    gs = GridSearchCV(classifier, grid, scoring=scoring, cv=k_fold, n_jobs=-1)
    gs.fit(X_train, t_train)

    print(gs.best_estimator_)
    use_model(gs, gs.best_score_, X_test,
              t_test, 'Logistic Regression')


def use_model(model, val_acc, x_test, t_test, name):
    print(f'Model: {name}')
    print()

    print('Training Performance')
    print(f'-> Acc: {val_acc}')

    print('Testing Performance')
    print(f'-> Acc: {accuracy_score(t_test, model.predict(x_test))}')
    print()


##############################
# Main code starts here
N_class = 3        # 0...2

alpha = 0.1         # learning rate
batch_size = 10    # batch size
MaxIter = 100        # Maximum iteration
decay = 0.          # weight decay

random_seed = 314
scoring = 'accuracy'

X, t = read_iris_data()

# Shape debugging
# print("X, t")
# print(X.shape, t.shape)

X_train, X_test, t_train, t_test = train_test_split(
    X, t, test_size=0.33, random_state=random_seed)

print('There are {} samples in the training set and {} samples in the test set'.format(
    X_train.shape[0], X_test.shape[0]))


# Baseline - Majority guess
baseline_prediction(X_train, X_test, t_train, t_test)

# Linear Classifier
logistic_regression_prediction(X_train, X_test, t_train, t_test)

# Non-linear Classifiers
