# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 11:48:11 2018

@author: Pawan
"""

import datetime
StartTime = datetime.datetime.now().replace(microsecond=0)
from dataPrep import *
from DbUtilities import *
from time import strftime
#from ClassficationModels import * 
from model_utility import * 
from TextClassModels import*


###################### reading from csv  ############################################################
#
#file_name = "SMART_UKR_Jan2018_data_CSV.csv"   #file_name = "TestData_SMARTReport_JUN.xlsx"

#df = conv_excl_to_df(file_name)


##################################################################################################
#df = silce_df(df,2,4)

ConnParam = load_DBconfig()
DbConn = conn(ConnParam['hostname'],ConnParam['port'],ConnParam['username'],ConnParam['password'],ConnParam['schema'])

SQLquery = 'SELECT ticket_assigned,ticket_desc FROM smart_machine.'+ConnParam['source']
df = dataframe_from_db(SQLquery,DbConn) 




df = data_cleaning(df)


#df.to_csv('data_frame_raw.csv',sep =',')

#df['text_lemmatized'] = df.Email.apply(lemmatize_text)
#df.text_lemmatized = df.text_lemmatized.apply(StopWords)
     
X,y = creating_datset_labels(df,1,0)

#X,vectorizer = feature_extarction_tfidf(X)


#save_model(vectorizer,"vectorizer")
myDict = {}

X_train, X_test, Y_train, Y_test  = data_spilt(X,y,0.3,42)


myDict['build_id'] = int(strftime("%m%d%Y%H%M"))
MultinomialNB_clf,MultinomialNB_accuracy = MultinomialNB_implmenation(X_train, X_test, Y_train, Y_test)
print('MultinomialNB_clf ----> ',MultinomialNB_accuracy)
MultinomialNB_accuracy = round(MultinomialNB_accuracy*100,2)
myDict['MultinomialNB_clf'] = MultinomialNB_accuracy
save_model(MultinomialNB_clf,"MultinomialNB")


BernoulliNB_clf,BernoulliNB_accuracy = BernoulliNB_implmenation(X_train, X_test, Y_train, Y_test)
print('BernoulliNB_clf ----> ',BernoulliNB_accuracy)
BernoulliNB_accuracy = round(BernoulliNB_accuracy*100,2)
myDict['BernoulliNB_clf'] = BernoulliNB_accuracy
save_model(BernoulliNB_clf,"BernoulliNB")


LinearSVC_clf,LinearSVC_accuracy = LinearSVC_implmenation(X_train, X_test, Y_train, Y_test)
print('LinearSVC_clf ----> ',LinearSVC_accuracy)
LinearSVC_accuracy = round(LinearSVC_accuracy*100,2)
myDict['LinearSVC_clf'] = LinearSVC_accuracy
save_model(LinearSVC_clf,"LinearSVC")


SGDClassifier_clf,SGDClassifier_accuracy = SGDClassifier_implmenation(X_train, X_test, Y_train, Y_test)
print('SGDClassifier_clf ----> ',SGDClassifier_accuracy)
SGDClassifier_accuracy = round(SGDClassifier_accuracy*100,2)
myDict['SGDClassifier_clf'] = SGDClassifier_accuracy
save_model(SGDClassifier_clf,"SGDClassifier")


LogisticRegression_clf,LogisticRegression_accuracy = LogisticRegression_implmenation(X_train, X_test, Y_train, Y_test)
print('LogisticRegression_clf ----> ',LogisticRegression_accuracy)
LogisticRegression_accuracy = round(LogisticRegression_accuracy*100,2)
myDict['LogisticRegression_clf'] = LogisticRegression_accuracy
save_model(LogisticRegression_clf,"LogisticRegression")


RandomForestClassifier_clf,RandomForestClassifier_accuracy = RandomForestClassifier_implmenation(X_train, X_test, Y_train, Y_test)
print('RandomForestClassifier_clf ----> ',RandomForestClassifier_accuracy)
RandomForestClassifier_accuracy = round(RandomForestClassifier_accuracy*100,2)
myDict['RandomForestClassifier_clf'] = RandomForestClassifier_accuracy
save_model(RandomForestClassifier_clf,"RandomForestClassifier")

insert_dict_to_db(myDict,DbConn,ConnParam['build_stats'],StartTime)

DbConn.close()