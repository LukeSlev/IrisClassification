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
# sklearn
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
# keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier


# Data viz
import seaborn as sns
import matplotlib.pyplot as plt


def read_iris_data():
    # Reading data from CSV file
    names = ['sepal_length', 'sepa_width',
             'petal_length', 'petal_width', 'class']
    parent_dir_path = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(
        parent_dir_path, 'input/bezdekIris.data'), names=names, engine='python')

    # scatter plot matrix
    # sns.pairplot(df, hue='class')

    # Check for any null values
    missing_df = df.isnull().sum()
    print("Nulls per column")
    print(missing_df)
    print()

    X = df.iloc[:, :4]
    t = df.loc[:, 'class']

    return X, t


def baseline_prediction(X_train, X_test, t_train, t_test, strategy='most_frequent'):
    classifier = DummyClassifier(strategy=strategy)
    classifier.fit(X_train, t_train)
    val_acc = classifier.score(X_train, t_train)

    use_model(classifier, val_acc, X_train, X_test, t_test, 'Majority Guess')


def logistic_regression_prediction(X_train, X_test, t_train, t_test):
    lrclassifier = LogisticRegression(random_state=random_seed, max_iter=10000)
    k_fold = StratifiedKFold(
        n_splits=folds, random_state=random_seed, shuffle=True)
    grid = {'C': np.power(10.0, np.arange(-10, 10))}

    gs = GridSearchCV(lrclassifier, grid, scoring=scoring,
                      cv=k_fold, n_jobs=-1)
    gs.fit(X_train, t_train)

    print(gs.best_estimator_)
    use_model(gs, gs.best_score_, X_train, X_test,
              t_test, 'Logistic Regression')


def naive_bayes_prediction(X_train, X_test, t_train, t_test):
    nbclassifier = GaussianNB()
    nbclassifier.fit(X_train, t_train)
    use_model(nbclassifier, nbclassifier.score(X_test, t_test), X_train, X_test,
              t_test, 'Gaussian Naïve Bayes Regression')


def svm_prediction(X_train, X_test, t_train, t_test):
    svclassifier = SVC(kernel='rbf', random_state=random_seed)
    k_fold = StratifiedKFold(
        n_splits=folds, random_state=random_seed, shuffle=True)

    param_grid = [
        {'C': np.power(10.0, np.arange(-10, 10)), 'kernel': ['linear']}, {'C': np.power(
            10.0, np.arange(-10, 10)), 'gamma': np.power(10.0, np.arange(-10, 10)), 'kernel': ['rbf']}
    ]

    gs = GridSearchCV(svclassifier, param_grid,
                      scoring=scoring, cv=k_fold, n_jobs=-1)
    gs.fit(X_train, t_train)

    print(gs.best_estimator_)
    use_model(gs, gs.best_score_, X_train, X_test,
              t_test, 'SVM Classification')


def knn_prediction(X_train, X_test, t_train, t_test):
    knnclassifier = KNeighborsClassifier()
    k_fold = StratifiedKFold(
        n_splits=folds, random_state=random_seed, shuffle=True)

    param_grid = {'n_neighbors': np.arange(1, 25)}

    gs = GridSearchCV(knnclassifier, param_grid,
                      scoring=scoring, cv=k_fold, n_jobs=-1)
    gs.fit(X_train, t_train)

    print(gs.best_params_)
    use_model(gs, gs.best_score_, X_train, X_test,
              t_test, 'KNN Classification')


def neural_network_classifier(X_train, X_test, t_train, t_test):
    '''4 features in the input layer (the four flower measurements), 3 classes in the ouput layer
    (corresponding to the 3 species), and 16 hidden units because
    (from the point of view of a GPU, 16 is a round number!)'''
    # Function to create model, required for KerasClassifier
    def create_model():
        # create model
        model = Sequential()
        model.add(Dense(10, input_shape=(4,), activation='tanh'))
        model.add(Dense(8, activation='tanh'))
        model.add(Dense(6, activation='tanh'))
        model.add(Dense(3, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
        return model

    nnclassifier = KerasClassifier(build_fn=create_model, verbose=0)

    k_fold = StratifiedKFold(
        n_splits=folds, random_state=random_seed, shuffle=True)

    param_grid = {
        "batch_size": [10, 20, 40, 60, 80, 100],
        "epochs": [10, 50, 100]
    }

    gs = GridSearchCV(nnclassifier, param_grid,
                      scoring=scoring, cv=k_fold, n_jobs=-1)
    gs.fit(X_train, t_train)

    print(gs.best_estimator_)
    use_model(gs, gs.best_score_, X_train, X_test,
              t_test, 'Neural Net Classification')


def use_model(model, val_acc, x_train, x_test, t_test, name):

    print(f'Model: {name}')
    print()

    print('Training Performance')
    print(f'-> Acc: {val_acc}')

    print('Testing Performance')
    print(f'-> Acc: {accuracy_score(t_test, model.predict(x_test))}')
    print()

    print('Report')
    evaluate_model(t_test, model.predict(x_test))
    print()


def evaluate_model(t_test, t_pred):
    print(confusion_matrix(t_test, t_pred))
    print(classification_report(t_test, t_pred))


def one_hot_encode_object_array(arr):
    '''One hot encode a numpy array of objects (e.g. strings)'''
    uniques, ids = np.unique(arr, return_inverse=True)
    return np_utils.to_categorical(ids, len(uniques))


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

X_scaled = StandardScaler().fit_transform(X)
X_train, X_test, t_train, t_test = train_test_split(
    X_scaled, t, test_size=0.33, random_state=random_seed, stratify=t)

print('There are {} samples in the training set and {} samples in the test set'.format(
    X_train.shape[0], X_test.shape[0]))

folds = 20
print(f'Cross validation with {folds} folds, random_seed {random_seed}')
print()


# Baseline - Majority guess
baseline_prediction(X_train, X_test, t_train, t_test)

# Linear Classifier
logistic_regression_prediction(X_train, X_test, t_train, t_test)

# Non-linear Classifiers
naive_bayes_prediction(X_train, X_test, t_train, t_test)
svm_prediction(X_train, X_test, t_train, t_test)
knn_prediction(X_train, X_test, t_train, t_test)  # non parametric
neural_network_classifier(X_train, X_test, t_train, t_test)
