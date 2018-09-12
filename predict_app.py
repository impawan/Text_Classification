# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 10:34:12 2018

@author: paprasad
"""

import os
from time import strftime
import numpy as np
import pandas as pd
import pickle
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
from statistics import mode
import re




def predict(text):
    pattern = re.compile('[\W_]+')
    text = pattern.sub('', text)   
    text = text.lower().strip('\n')
    text = [text]
    
    
    file_object = open('./saved_models/MultinomialNB.pickle','rb')
    MultinomialNB_clf = pickle.load(file_object)
    
#    file_object = open('./saved_models/BernoulliNB.pickle','rb')
#    BernoulliNB_clf = pickle.load(file_object)
#        
#        
    file_object = open('./saved_models/SGDClassifier.pickle','rb')
    SGDClassifier_clf = pickle.load(file_object)
        
    file_object = open('./saved_models/LogisticRegression.pickle','rb')
    LogisticRegression_clf = pickle.load(file_object)
        
    file_object = open('./saved_models/LinearSVC.pickle','rb')
    LinearSVC_clf = pickle.load(file_object)

    file_object = open('./saved_models/RandomForestClassifier.pickle','rb')
    RandomForestClassifier_clf = pickle.load(file_object)
    vote = []
    try:
       
        temp = MultinomialNB_clf.predict(text)
        vote.append(temp[0])
        
#        temp = BernoulliNB_clf.predict(text)
#        vote.append(temp[0])
        
        
        temp = SGDClassifier_clf.predict(text)
        vote.append(temp[0])
        
        
        temp = LogisticRegression_clf.predict(text)
        vote.append(temp[0])
        
        temp = LinearSVC_clf.predict(text)
        vote.append(temp[0]) 
        
        temp = RandomForestClassifier_clf.predict(text)
        vote.append(temp[0]) 
        
        
        ret = mode(vote)
    except Exception as error:
        ret = str(error)+" i.e  "+str(vote)
        pass
            
    return str(ret)