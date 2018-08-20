# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 15:47:57 2018

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

file_name = "TestData_SMARTReport_JUN.xlsx"
df = pd.read_excel(file_name)
df = df.iloc[:,2:4]
 

stop_words = nltk.corpus.stopwords.words('english')
newStopWords = ['Regards','Thanks','Please','find',':','(',')',';','<','@','?','{','}','/']
stop_words.extend(newStopWords)


#data cleaning, removing new line. commas and semicolon
df = df.replace(r'\r',' ', regex=True) 
df = df.replace(r'\t',' ', regex=True)
df = df.replace(r'\n',' ', regex=True) 
df = df.replace(r';',' ', regex=True) 
df = df.replace(r',',' ', regex=True)
df = df.replace(r'[^a-zA-Z\d\s]','',regex=True)
df = df.replace(r'(<!--((.|\\R)*)-->)',' ',regex=True)
#df = df.replace(r'(<!--((.|R)*)-->)',' ')


#this will remove the null rows
df = df.dropna(how='any',axis = 0)
df.columns = ['Email_body','Assigned']
df.Email_body = df.Email_body.apply(StopWords)
print("--------------------------unique lables in data-----------------------------\n\n\n\n")

print(df.Assigned.unique())



def StopWords(text):
    ret = ''
    for word in word_tokenize(text):
        word = word.lower()
        if word not in stop_words:
            ret = ret+' '+word
    return ret  



def predict(email_body):
    
    vote = []
    try:
       
        temp = MultinomialNB_clf.predict(email_body)
        vote.append(temp[0])
        
        temp = BernoulliNB_clf.predict(email_body)
        vote.append(temp[0])
        
        
        temp = SGDClassifier_clf.predict(email_body)
        vote.append(temp[0])
        
        
        temp = LogisticRegression_clf.predict(email_body)
        vote.append(temp[0])
        
        temp = LinearSVC_clf.predict(email_body)
        vote.append(temp[0]) 
        ret = mode(vote)
    except Exception as error:
        ret = str(error)+" i.e  "+str(vote)
        pass
            
    return ret


#def load_model():
#    
#    file_object = open('MultinomialNB.pickle','rb')
#    MultinomialNB_clf = pickle.load(file_object)
#    
#    file_object = open('BernoulliNB.pickle','rb')
#    BernoulliNB_clf = pickle.load(file_object)
#        
#        
#    file_object = open('SGDClassifier.pickle','rb')
#    SGDClassifier_clf = pickle.load(file_object)
#        
#    file_object = open('LogisticRegression.pickle','rb')
#    LogisticRegression_clf = pickle.load(file_object)
#        
#    file_object = open('LinearSVC.pickle','rb')
#    LinearSVC_clf = pickle.load(file_object)
    
    
#load_model()
    

file_object = open('MultinomialNB.pickle','rb')
MultinomialNB_clf = pickle.load(file_object)
    
file_object = open('BernoulliNB.pickle','rb')
BernoulliNB_clf = pickle.load(file_object)
        
        
file_object = open('SGDClassifier.pickle','rb')
SGDClassifier_clf = pickle.load(file_object)
        
file_object = open('LogisticRegression.pickle','rb')
LogisticRegression_clf = pickle.load(file_object)
        
file_object = open('LinearSVC.pickle','rb')
LinearSVC_clf = pickle.load(file_object)

predicted = []
for email in df['Email_body']:
    email_body = []
    email_body.append(email) 
    ret = predict(email_body)
    #print (ret)
    predicted.append(ret)

 

df['predicted']=predicted

result = np.mean(df['Assigned']==df['predicted'])*100
print ('------------------------------> Result',result)
df['result']=''
df.at[0, 'result'] = result

#directory = '/D:/Profiles/paprasad/python/Text Classification/Bug_triager/test_result/'
filename = 'Result'+strftime("%m%d%Y%H%M")+'___'+str(result)+'______'+'.csv'
#file = open(os.path.join(directory, filename), "w")

df.to_csv(filename, sep=',')







def spilt_test_prediction(df):
    test_results =[]
    last_index = 0
    for index in range (200,df.size,200):
        print (last_index,'-----------',index)
        temp_df = df.loc[last_index:index,:]
#        print (temp_df.shape)
#        print(type(temp_df))
#        print(type(df))
        #print('------------------------------------------>',temp_df.size,'---------->',index)
        last_index=index
        predicted = []
#        print (temp_df['Email body'])
        for email in temp_df['Email body']:
            email_body = []
            email_body.append(email)
            ret = predict(email_body)
    #print (ret)
            predicted.append(ret)
        temp_df['predicted']=predicted
            

        result = np.mean(temp_df['Assigned']==temp_df['predicted'])*100
        print ('------------------------------> Result',result)
   
    

#spilt_test_prediction(df)
