# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 12:14:47 2018

@author: Pawan
"""
import numpy as np
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC


def MultinomialNB_implmenation(X_train, X_test, Y_train, Y_test,alpha = 1.0, fit_prior = True, class_prior = None):
    MultinomialNB_clf = MultinomialNB(alpha = alpha, fit_prior = fit_prior, class_prior = class_prior)
    MultinomialNB_clf = MultinomialNB_clf.fit(X_train,Y_train)
    predicted = MultinomialNB_clf.predict(X_test)
    accuracy =  np.mean(predicted == Y_test)
    return MultinomialNB_clf,accuracy



def BernoulliNB_implmenation(X_train, X_test, Y_train, Y_test,alpha = 1.0, fit_prior = True, class_prior = None):
    BernoulliNB_clf = BernoulliNB(alpha = alpha, fit_prior = fit_prior, class_prior = class_prior)
    BernoulliNB_clf = BernoulliNB_clf.fit(X_train,Y_train)
    predicted = BernoulliNB_clf.predict(X_test)
    accuracy =  np.mean(predicted == Y_test)
    return BernoulliNB_clf,accuracy


def SGDClassifier_implmenation(X_train, X_test, Y_train, Y_test):
    SGDClassifier_clf = SGDClassifier()
    SGDClassifier_clf = SGDClassifier_clf.fit(X_train,Y_train)
    predicted = SGDClassifier_clf.predict(X_test)
    accuracy =  np.mean(predicted == Y_test)
    return SGDClassifier_clf,accuracy
    
def LogisticRegression_implmenation(X_train, X_test, Y_train, Y_test):
    LogisticRegression_clf = LogisticRegression()
    LogisticRegression_clf = LogisticRegression_clf.fit(X_train,Y_train)
    predicted = LogisticRegression_clf.predict(X_test)
    accuracy =  np.mean(predicted == Y_test)
    return LogisticRegression_clf,accuracy

def LinearSVC_implmenation(X_train, X_test, Y_train, Y_train):
    LinearSVC_clf = LinearSVC()
    LinearSVC_clf = LinearSVC_clf.fit(X_train,Y_test)
    predicted = LinearSVC_clf.predict(X_test)
    accuracy =  np.mean(predicted == Y_test)
    return LinearSVC_clf,accuracy

def RandomForestClassifier_impmenation(X_train, X_test, Y_train, Y_test):
    RandomForest_clf = RandomForestClassifier()
    RandomForest_clf = RandomForest_clf.fit(X_train,Y_train)
    predicted = RandomForest_clf.predict(X_test)
    accuracy =  np.mean(predicted == Y_test)
    return RandomForest_clf, accuracy
    
    