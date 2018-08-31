# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 11:48:11 2018

@author: Pawan
"""

from dataPrep import *
from ClassficationModels import * 
from model_utility import * 

file_name = "SMART_UKR_Jan2018_data_CSV.csv"
#file_name = "TestData_SMARTReport_JUN.xlsx"
df = conv_excl_to_df(file_name)

#df = silce_df(df,2,4)

df = data_cleaning(df)


df.to_csv('data_frame_raw.csv',sep =',')

#df['text_lemmatized'] = df.Email.apply(lemmatize_text)
#df.text_lemmatized = df.text_lemmatized.apply(StopWords)
     
X,y = creating_datset_labels(df,1,0)

X,vectorizer = feature_extarction_tfidf(X)

save_model(vectorizer,"vectorizer")


X_train, X_test, Y_train, Y_test  = data_spilt(X,y,0.3,42)

MultinomialNB_clf,MultinomialNB_accuracy = MultinomialNB_implmenation(X_train, X_test, Y_train, Y_test)
print('MultinomialNB_clf ----> ',MultinomialNB_accuracy)
save_model(MultinomialNB_clf,"MultinomialNB")

BernoulliNB_clf,BernoulliNB_accuracy = BernoulliNB_implmenation(X_train, X_test, Y_train, Y_test)
print('BernoulliNB_clf ----> ',BernoulliNB_accuracy)
save_model(BernoulliNB_clf,"BernoulliNB")


LinearSVC_clf,LinearSVC_accuracy = LinearSVC_implmenation(X_train, X_test, Y_train, Y_test)
print('LinearSVC_clf ----> ',LinearSVC_accuracy)
save_model(LinearSVC_clf,"LinearSVC")

SGDClassifier_clf,SGDClassifier_accuracy = SGDClassifier_implmenation(X_train, X_test, Y_train, Y_test)
print('SGDClassifier_clf ----> ',SGDClassifier_accuracy)
save_model(SGDClassifier_clf,"SGDClassifier")

LogisticRegression_clf,LogisticRegression_accuracy = LogisticRegression_implmenation(X_train, X_test, Y_train, Y_test)
print('LogisticRegression_clf ----> ',LogisticRegression_accuracy)
save_model(LogisticRegression_clf,"LogisticRegression")


RandomForestClassifier_clf,RandomForestClassifier_accuracy = RandomForestClassifier_impmenation(X_train, X_test, Y_train, Y_test)
print('RandomForestClassifier_clf ----> ',RandomForestClassifier_accuracy)
save_model(RandomForestClassifier_clf,"RandomForestClassifier")