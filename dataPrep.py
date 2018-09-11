#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 11:48:07 2018

@author: pawan
"""



import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer
from nltk.tokenize import sent_tokenize, word_tokenize


from sklearn.model_selection import train_test_split


ps = PorterStemmer()


def conv_to_df(file):
    '''
    Method for converting the input .csv file into data frame. 
    '''
    df = pd.read_csv(file,encoding='latin-1')
    return df


def conv_excl_to_df(file):
    '''
    Method for converting Excel sheet into data frame
    '''
    xl = pd.ExcelFile(file)
    sheet = xl.sheet_names
    sheet = ''.join(sheet)
    df = xl.parse(sheet)
    return df



def silce_df(df,col1,col2):
    return df.iloc[:,col1:col2]

lemmatizer = nltk.stem.WordNetLemmatizer()
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()

stop_words = nltk.corpus.stopwords.words('english')
newStopWords = ['Regards','Thanks','Please','find','warm']
stop_words.extend(newStopWords)

def data_cleaning(df):
    '''
    This methdo cleans the data for \n,\r and drop null coumns rows
    define column for data frame
    also apply lemmatizeation on the data frame columns 
    '''
    
    df = df.apply(lambda x: x.astype(str).str.lower())

    df = df.replace(r'(<!--)(.*)(-->)','',regex=True)
    df = df.replace(r'\n',' ', regex=True) 
    df = df.replace(r'\r',' ', regex=True) 
    df = df.replace(r'\t',' ', regex=True) 
    df =  df.replace(',',' ') 
    df = df.replace(r'\\n',' ', regex=True)
    df = df.replace(r'&','',regex=True)
    df = df.replace(r'#','',regex=True)
    
    df= df.replace(r'span((.|\n)*)\}',' ',regex= True)
    df = df.replace(r'@font-face((.|\n)*)\}',' ',regex = True)
    
    df = df.dropna(how='any',axis=0)
    df.columns = ['Assignee', 'Email']
    df.Email = df.Email.str.lower()
    df = df.replace(r';',' ', regex=True)
    df['text_lemmatized'] = df.Email.apply(lemmatize_text)
    df.text_lemmatized = df.Email.apply(StopWords)
    return df


def lemmatize_text(text):
    '''
    This method applies lemmatizeation on the input text value
    '''
    temp=''
    for w in w_tokenizer.tokenize(text):
        temp = temp+' '+lemmatizer.lemmatize(w)
        temp = temp.lower()
    return temp

def text_stemming(text):
    '''
    This method applies stemming on the input text value
    '''
    temp=''
    #print(text)
    for w in w_tokenizer.tokenize(text):
        temp = temp+' '+ ps.stem(w)
    return temp   

def creating_datset_labels(df,X_col_index, Y_col_index):
    '''
    Create X,y lables from training data
    '''
    X = df.iloc[:, X_col_index].values
    Y = df.iloc[:, Y_col_index].values
    return X,Y

def visualise(df,feature_name):
    '''
    visualise the input column data with histogram
    Note -- I need to more modification to this.
    '''
    plt.hist(df[feature_name],)









def getUnique(df, feature_name):
    
    '''
    Return the unique values within data frame for specified input column value.
    '''
    return df.Assignee.unique()


def feature_extarction_tfidf(X):
    
    '''
    This method ext
    '''
    vectorizer = TfidfVectorizer(min_df=1,lowercase=True,stop_words=stop_words)
    feature_mat = vectorizer.fit_transform(X)
    idf = vectorizer.idf_
    feature_weights = dict(zip(vectorizer.get_feature_names(), idf))
    return feature_mat,vectorizer


def categrorical_data_enc(Y):
    '''
    Method for label encoding 
    '''
    encoder = LabelBinarizer()
    encoded_label = encoder.fit_transform(Y)
    return encoded_label


def GetFreqCount(x):
    '''
    This method return dictonary of word with frequencies
    '''
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
    '''
    This mehtod convert dictonary to list data type
    '''
    dictList = list()
    for key in dic.items():
        temp = [key,dic[key]]
        dictList.append(temp)
    return dictList


def ConvertDictToDataFrame(dic):
    '''
    Method for converting dictonary to data frame
    '''
    return pd.DataFrame.from_dict(dic)
    


def StopWords(text):
    '''
    This method suppress the stop words from a text.
    '''
    ret = ''
    for word in word_tokenize(text):
        word = word.lower()
        if word not in stop_words:
            ret = ret+' '+word
    return ret  



def data_spilt(X,y,test_ratio,rnd_state):
    '''
    Spilt the data into testing and training set
    '''
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=test_ratio, random_state=rnd_state)
    return  X_train, X_test, Y_train, Y_test
