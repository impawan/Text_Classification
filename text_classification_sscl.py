# -*- coding: utf-8 -*-
"""
Created on Fri May 11 09:04:41 2018

@author: paprasad
"""

import pandas as pd
import numpy as np


folder = 'd:\Profiles\paprasad\python\Text Classification\Bug Traiger'
file_name='\SMART_UKR_Jan2018_data_CSV.csv'
path = folder+file_name
xl = pd.ExcelFile(path)



def models(data):
    numpy_array = data.as_matrix()
    X = numpy_array[:,1]
    Y = numpy_array[:,0]
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(
     X, Y, test_size=0.4, random_state=52)
    
    from sklearn.feature_extraction.text import CountVectorizer
    
    from sklearn.feature_extraction.text import TfidfTransformer
    
    #different Model of ML
    
    from sklearn.naive_bayes import MultinomialNB  
    
    from sklearn.linear_model import SGDClassifier
    
    from sklearn.pipeline import Pipeline
    text_clf_NB = Pipeline([('vect', CountVectorizer(stop_words='english')),
     ('tfidf', TfidfTransformer()),
     ('clf', MultinomialNB()),
    ])
    
    
    text_clf_NB = text_clf_NB.fit(X_train.astype(str),Y_train.astype(str))
    
    predicted = text_clf_NB.predict(X_test.astype(str))
    accuracy = np.mean(predicted == Y_test)
    
    print ("Accurcy for Naive Bayes:  ",accuracy)
    
    text_clf_SVM = Pipeline([('vect', CountVectorizer(stop_words='english')),
     ('tfidf', TfidfTransformer()),
     ('clf', SGDClassifier()),
    ])
    
    
    
    text_clf_SVM = text_clf_SVM.fit(X_train.astype(str),Y_train.astype(str))
    
    predicted = text_clf_SVM.predict(X_test.astype(str))
    accuracy = np.mean(predicted == Y_test)
    
    print ("Accurcy for SVM:  ",accuracy)


def text_classification(sheet):
   df = xl.parse(sheet)
   df = df.replace(r'\n',' ', regex=True) 
#   df = df.replace(r'<',' ', regex=True) 
#   df = df.replace(r'>',' ', regex=True) 
   models(df)
   








for sheet in xl.sheet_names:
    table_name = sheet
    text_classification(sheet)
    
