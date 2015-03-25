#!/usr/bin/env python
# -*- coding: utf8 -*-
#
# Author: Alexice
# Date: 14.01.2012
#
#
import warnings

from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm.sparse import SVC as sSVC, NuSVC as sNuSVC
from sklearn.svm import SVC, NuSVC
#from sklearn.linear_model.sparse import SGDClassifier as sSGDClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier

warnings.filterwarnings("ignore")

learners_dense = [
    SVC(kernel="linear"),
    NuSVC(kernel="rbf"),
    SGDClassifier(),
    KNeighborsClassifier(n_neighbors=11, weights='distance'),
    BernoulliNB(),
    GradientBoostingClassifier()
]

learners_sparse = [
    sSVC(kernel="linear"),
    sNuSVC(kernel="rbf"),
    SGDClassifier(),
    KNeighborsClassifier(n_neighbors=11, weights='distance'),
    BernoulliNB(),
    GradientBoostingClassifier()
]

learner_names = [
    "Support Vector Machines Classifier (linear)",
    "Nu-Support Vector Machines Classifier (rbf)",
    "Linear Stochastic Gradient Descent Classifier",
    "k-nearest neighbors Classifier",
    "Naive Bayes Classifier (Bernoulli)",
    "Boosting Classifier",
    
    "Ensemble [majority]",
    "Ensemble [all & all but one]",
    "Ensemble [all]",

    "Always 'pos' Classifier [for camparison]",
    "Always 'neg' Classifier [for camparison]",
    "Random Classifier [for camparison]",
]
