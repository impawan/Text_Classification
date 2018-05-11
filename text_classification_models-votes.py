# -*- coding: utf-8 -*-
"""
Created on Fri May 11 12:31:35 2018

@author: paprasad
"""

import nltk
import pandas as pd
import numpy as np


folder = 'd:\Profiles\paprasad\python\Text Classification\Bug Traiger'
file_name='\SMART_UKR_Jan2018_data_CSV.csv'
path = folder+file_name
xl = pd.ExcelFile(path)



from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

def model(data):
    
    numpy_array = data.as_matrix()
    features  = numpy_array[:,1]
    lables  = numpy_array[:,0]
    #document = list(zip(features, lables))
    
    #documents = dict(zip(features, lables))
    documents = [(features,lables)]
    print(type(documents))
    
    featuresets = [(features, lables) for (features, lables) in documents]
    
    # set that we'll train our classifier with
    training_set = featuresets[:1900]
    
    # set that we'll test against.
    testing_set = featuresets[1900:]
    
    print ("recahed here-----------------------------------------------")
    
    MNB_classifier = SklearnClassifier(MultinomialNB())
    MNB_classifier.train(training_set)
    print("MultinomialNB accuracy percent:",nltk.classify.accuracy(MNB_classifier, testing_set))
    BNB_classifier = SklearnClassifier(BernoulliNB())
    BNB_classifier.train(training_set)
    print("BernoulliNB accuracy percent:",nltk.classify.accuracy(BNB_classifier, testing_set))
    print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100)
    #classifier.show_most_informative_features(15)
        
    LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
    LogisticRegression_classifier.train(training_set)
    print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)
    
    SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
    SGDClassifier_classifier.train(training_set)
    print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)
    
    SVC_classifier = SklearnClassifier(SVC())
    SVC_classifier.train(training_set)
    print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)
    
    LinearSVC_classifier = SklearnClassifier(LinearSVC())
    LinearSVC_classifier.train(training_set)
    print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)
    
    NuSVC_classifier = SklearnClassifier(NuSVC())
    NuSVC_classifier.train(training_set)
    print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)
    
    
def text_classification(sheet):
   df = xl.parse(sheet)
   df = df.replace(r'\n',' ', regex=True) 
   #df = df.replace(r';',' ', regex=True) 
#   df = df.replace(r'>',' ', regex=True) 
   model(df)
   
   
   
   


for sheet in xl.sheet_names:
    table_name = sheet
    text_classification(sheet)   