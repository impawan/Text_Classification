# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 11:48:11 2018

@author: Pawan
"""

from dataPrep import *

file_name = "report_clean.csv"

df = conv_to_df(file_name)

df = data_cleaning(df)

df['text_lemmatized'] = df.Email.apply(lemmatize_text)
df.text_lemmatized = df.text_lemmatized.apply(StopWords)

X,y = creating_datset_labels(df,2,0)
X = feature_extaction_tfidf(X)

X_train, X_test, Y_train, Y_test  = data_spilt(X,y,0.3,42)

MultinomialNB_clf,MultinomialNB_accuracy = MultinomialNB_implmenation(X_train, X_test, Y_train, Y_test)