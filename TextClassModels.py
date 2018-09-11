# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 12:41:28 2018

@author: paprasad
"""





from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
import numpy as np



from sklearn.model_selection import GridSearchCV


parameters_svm = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False), 'clf__alpha': (1e-2, 1e-3),}

gs_clf_svm = GridSearchCV(SGDClassifier_clf, parameters_svm,n_jobs=1)
gs_clf_svm = gs_clf_svm.fit(X_train, Y_train)
gs_clf_svm.best_score_
gs_clf_svm.best_params_
test = gs_clf_svm.predict(X_train)
print(test)

def MultinomialNB_implmenation(X_train, X_test, Y_train, Y_test,alpha = 1.0, fit_prior = True, class_prior = None):
    MultinomialNB_clf = Pipeline([('vect', CountVectorizer(stop_words='english')), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB()) ])
    MultinomialNB_clf = MultinomialNB_clf.fit(X_train,Y_train)
    predicted = MultinomialNB_clf.predict(X_test)
    accuracy =  np.mean(predicted == Y_test)
    return MultinomialNB_clf, accuracy


def BernoulliNB_implmenation(X_train, X_test, Y_train, Y_test,alpha = 1.0, fit_prior = True, class_prior = None):
    BernoulliNB_clf = Pipeline([('vect', CountVectorizer(stop_words='english')), ('tfidf', TfidfTransformer()), ('clf', BernoulliNB()) ])
    BernoulliNB_clf = BernoulliNB_clf.fit(X_train,Y_train)
    predicted = BernoulliNB_clf.predict(X_test)
    accuracy =  np.mean(predicted == Y_test)
    return BernoulliNB_clf, accuracy


def SGDClassifier_implmenation(X_train, X_test, Y_train, Y_test):
    SGDClassifier_clf = Pipeline([('vect', CountVectorizer(stop_words='english')), ('tfidf', TfidfTransformer()), ('clf', SGDClassifier(loss='hinge')) ])
    SGDClassifier_clf = SGDClassifier_clf.fit(X_train,Y_train)
    predicted = SGDClassifier_clf.predict(X_test)
    accuracy =  np.mean(predicted == Y_test)
    return SGDClassifier_clf, accuracy


def LogisticRegression_implmenation(X_train, X_test, Y_train, Y_test):
    LogisticRegression_clf = Pipeline([('vect', CountVectorizer(stop_words='english')), ('tfidf', TfidfTransformer()), ('clf', LogisticRegression()) ])
    LogisticRegression_clf = LogisticRegression_clf.fit(X_train,Y_train)
    predicted = LogisticRegression_clf.predict(X_test)
    accuracy =  np.mean(predicted == Y_test)
    return LogisticRegression_clf, accuracy


def LinearSVC_implmenation(X_train, X_test, Y_train, Y_test):
    LinearSVC_clf = Pipeline([('vect', CountVectorizer(stop_words='english')), ('tfidf', TfidfTransformer()), ('clf', LinearSVC()) ])
    LinearSVC_clf = LinearSVC_clf.fit(X_train,Y_train)
    predicted = LinearSVC_clf.predict(X_test)
    accuracy =  np.mean(predicted == Y_test)
    return LinearSVC_clf, accuracy



def RandomForestClassifier_implmenation(X_train, X_test, Y_train, Y_test):
    RandomForestClassifier_clf = Pipeline([('vect', CountVectorizer(stop_words='english')), ('tfidf', TfidfTransformer()), ('clf', RandomForestClassifier()) ])
    RandomForestClassifier_clf = RandomForestClassifier_clf.fit(X_train,Y_train)
    predicted = RandomForestClassifier_clf.predict(X_test)
    accuracy =  np.mean(predicted == Y_test)
    return RandomForestClassifier_clf, accuracy



