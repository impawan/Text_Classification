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
from DbUtilities import *




def predict(text):
    
    ConnParam = load_DBconfig()
    DbConn = conn(ConnParam['hostname'],ConnParam['port'],ConnParam['username'],ConnParam['password'],ConnParam['schema'])
    ScoreDict = fetch_score_from_db()
    pattern = re.compile('[\W_]+')
    text = pattern.sub('', text)   
    text = text.lower().strip('\n')
    text = [text]
    VoteDict = {}
    
    file_object = open('./saved_models/MultinomialNB.pickle','rb')
    MultinomialNB_clf = pickle.load(file_object)
    

    file_object = open('./saved_models/BernoulliNB.pickle','rb')
    BernoulliNB_clf = pickle.load(file_object)
    
    
    
        
    file_object = open('./saved_models/SGDClassifier.pickle','rb')
    SGDClassifier_clf = pickle.load(file_object)
        
    file_object = open('./saved_models/LogisticRegression.pickle','rb')
    LogisticRegression_clf = pickle.load(file_object)
        
    file_object = open('./saved_models/LinearSVC.pickle','rb')
    LinearSVC_clf = pickle.load(file_object)

    file_object = open('./saved_models/RandomForestClassifier.pickle','rb')
    RandomForestClassifier_clf = pickle.load(file_object)
           
    temp = MultinomialNB_clf.predict(text)
    VoteDict['MultinomialNB_clf'] = temp
        
    temp = BernoulliNB_clf.predict(text)
    VoteDict['BernoulliNB_clf'] = temp
        
        
    temp = SGDClassifier_clf.predict(text)
    VoteDict['SGDClassifier_clf'] = temp
        
        
    temp = LogisticRegression_clf.predict(text)
    VoteDict['LogisticRegression_clf'] = temp
        
    temp = LinearSVC_clf.predict(text)
    VoteDict['LinearSVC_clf'] = temp
        
    temp = RandomForestClassifier_clf.predict(text)
    VoteDict['RandomForestClassifier_clf'] = temp 
        
        
    ret = mode(vote)
    
    return str(ret)