# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 10:09:59 2018

@author: paprasad
"""

import nltk
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from collections import Counter
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

import re

from scipy.sparse.csr import csr_matrix #need this if you want to save tfidf_matrix

from nltk.stem import PorterStemmer
ps = PorterStemmer()



def lemmatize_text(text):
    temp=''
    for w in w_tokenizer.tokenize(text):
        temp = temp+' '+lemmatizer.lemmatize(w)
    return temp


#folder = 'D:\Profiles\paprasad\python\Text Classification\Bug Traiger'
#file_name='/SMART_UKR_Jan2018_data_CSV.csv'
#path = folder+file_name
xl = pd.read_csv("report_clean.csv",encoding='utf-8')

stop_words = nltk.corpus.stopwords.words('english')
newStopWords = ['Regards','Thanks','Please','find','warm']
stop_words.extend(newStopWords)

def text_classification(df):
   
   df = df.replace(r'\n',' ', regex=True) 
   df = df.replace(r'\r',' ', regex=True) 
   df = df.replace(r'\t',' ', regex=True) 
   df =  df.replace(',',' ') 
  
   df = df.dropna(how='any',axis=0)
   #df = df.replace(r'\\n',' ', regex=True) 
   df.columns = ['Assignee', 'Email']
   #plt.hist(df['Assignee'],)
   df['text_lemmatized'] = df.Email.apply(lemmatize_text)
   df = df.replace(r';',' ', regex=True) 
   print("-------------------------unique lables ----------------------------------\n\n\n")
   print(df.Assignee.unique())
   return df


class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])
stemmed_count_vect = StemmedCountVectorizer(stop_words=stop_words)

df = text_classification(xl) 
#corpus = [df.iloc[:,2]]

corpus = []
for row in range(0,df.shape[0]):
    temp_string = df.iloc[row,2]
    #temp_string=re.sub('[^a-zA-Z]', ' ' ,temp_string) 
    #temp_string=[ps.stem(word) for word in temp_string if not word in set(stopwords.words('english'))]
    temp_string = temp_string.split(' ')
    stemmed_str = []
    for word in temp_string:
        word = ps.stem(word)
        stemmed_str.append(word)
        
    temp_string =  stemmed_str   
    temp_string = ' '.join(temp_string)
    corpus.append(temp_string)
    print (temp_string)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(stop_words=stop_words)
X = cv.fit_transform(corpus).toarray() 
MultinomialNB_clf = Pipeline([('vect', stemmed_count_vect),('tfidf', TfidfTransformer())])
X = MultinomialNB_clf.fit_transform(corpus)
temp = X.toarray()
print(temp.shape)
    
    
vectorizer = TfidfVectorizer(min_df=1,lowercase=True,stop_words=stop_words)
X1 = vectorizer.fit_transform(corpus)
idf = vectorizer.idf_
print (dict(zip(vectorizer.get_feature_names(), idf)))


'''
rows, cols = np.nonzero(temp)
print(temp[row,cols])
'''

#
#for row in range(len(df.index)-1):
#    temp = df['text_lemmatized'][row]
#    print('---------------------------->',row)
#    print(temp)

#
#import nltk
#import random
#from nltk.corpus import movie_reviews
#df = text_classification(sheet) 
#documents = [(list(df['text_lemmatized']), category)
#             for category in movie_reviews.categories()
#             for fileid in movie_reviews.fileids(category)]
#
#random.shuffle(documents)
#
#all_words = []
#
#for w in movie_reviews.words():
#    all_words.append(w.lower())
#
#all_words = nltk.FreqDist(all_words)