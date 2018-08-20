#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 11:48:07 2018

@author: pawan
"""



import nltk
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer
from nltk.tokenize import sent_tokenize, word_tokenize


ps = PorterStemmer()


def conv_pdf(file):
    df = pd.read_csv(file,encoding='utf-8')
    return df

lemmatizer = nltk.stem.WordNetLemmatizer()
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()

stop_words = nltk.corpus.stopwords.words('english')
newStopWords = ['Regards','Thanks','Please','find','warm']
stop_words.extend(newStopWords)

def data_cleaning(df):
   
   df = df.replace(r'\n',' ', regex=True) 
   df = df.replace(r'\r',' ', regex=True) 
   df = df.replace(r'\t',' ', regex=True) 
   df =  df.replace(',',' ') 
   df = df.replace(r'\\n',' ', regex=True)
   df = df.dropna(how='any',axis=0)
   df.columns = ['Assignee', 'Email']
   df = df.replace(r';',' ', regex=True)
   df['text_lemmatized'] = df.Email.apply(lemmatize_text)
   df.text_lemmatized = df.Email.apply(text_stemming)
   #plt.hist(df['Assignee'],)
   #df['text_lemmatized'] = df.Email.apply(lemmatize_text) 
   print("-------------------------unique lables ----------------------------------\n\n\n")
   print(df.Assignee.unique())
   return df


def lemmatize_text(text):
    temp=''
    for w in w_tokenizer.tokenize(text):
        temp = temp+' '+lemmatizer.lemmatize(w)
    return temp

def text_stemming(text):
    temp=''
    #print(text)
    for w in w_tokenizer.tokenize(text):
        temp = temp+' '+ ps.stem(w)
    return temp   

def creating_datset_labesl(df,X_col_index, Y_col_index):
    X = df.iloc[:, X_col_index].values
    Y = df.iloc[:, Y_col_index].values
    return X,Y

def visualise(df,feature_name):
    plt.hist(df[feature_name],)


def getUnique(df, feature_name):
    return df.Assignee.unique()


def feature_extaction(X):
    vectorizer = TfidfVectorizer(min_df=1,lowercase=True,stop_words=stop_words)
    feature_mat = vectorizer.fit_transform(X)
    idf = vectorizer.idf_
    feature_weights = dict(zip(vectorizer.get_feature_names(), idf))
    return feature_mat


def categrorical_data_enc(Y):
    encoder = LabelBinarizer()
    encoded_label = encoder.fit_transform(Y)
    return encoded_label


def GetFreqCount(x):
    word_cnt = {}
    for row in x.ix[:,0]:
        
        for word in word_tokenize(row) :
           # print(word)
            if word in word_cnt:
                word_cnt[word] = word_cnt[word]+1
                
            else:
                word_cnt[word] = 1
    return  word_cnt          
    

def ConvertDictToList(dic):
    dictList = list()
    for key in dic.items():
        temp = [key,dic[key]]
        dictList.append(temp)
    return dictList


def ConvertDictToDataFrame(dic):
    return pd.DataFrame.from_dict(dic)
    


def StopWords(text):
    ret = ''
    for word in word_tokenize(text):
        word = word.lower()
        if word not in stop_words:
            ret = ret+' '+word
    return ret  



df = conv_pdf("report_clean.csv")

df = data_cleaning(df)
X,Y = creating_datset_labesl(df,1,0)
X_train, X_test,Y_train,Y_test = train_test_split(X,Y, test_size = 0.2,random_state =0)

X_feature_tr = feature_extaction(X_train)

Y_enc = categrorical_data_enc(Y_train)