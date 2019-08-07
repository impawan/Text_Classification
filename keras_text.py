# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 14:12:18 2019

@author: paprasad
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
#from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt

import nltk
from nltk.stem.porter import PorterStemmer
porter_stemmer  = PorterStemmer()
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
from sklearn.preprocessing import LabelBinarizer

#import keras
#
##import keras.models
##from keras.models import Sequential
#from keras.layers import Activation, Dense, Dropout



import tensorflow as tf
from tensorflow import keras


import tensorflow as tf




df = pd.read_csv('./Dataset/train.csv')




df.head()




df.info()




from wordcloud import WordCloud, STOPWORDS

def word_cloud(data,color='black'):
    wordcloud = WordCloud(background_color=color, stopwords=STOPWORDS, max_words=200, max_font_size=40,  random_state=23)
    wordcloud = wordcloud.generate(str(data))
    fig = plt.figure(1, figsize=(20, 20))
    plt.imshow(wordcloud)
    
def stem(data):
    tokenization = nltk.word_tokenize(data)
    sentence = ''
    for w in tokenization:
        w= porter_stemmer.stem(w)
        sentence  = sentence+' '+w
    return sentence

def lemtize(data):
    tokenization = nltk.word_tokenize(data)
    sentence =''
    for w in tokenization:
        w = wordnet_lemmatizer.lemmatize(w)
        sentence = sentence+' '+w       
    return sentence
    




df['Review Text'] = df['Review Text'].apply(stem)
df['Review Text'] = df['Review Text'].apply(stem)
df['Review Title'] = df['Review Text'].apply(lemtize)
df['Review Title'] = df['Review Text'].apply(lemtize)





#word_cloud(train['Review Text'])





x = df[['Review Text','Review Title']]





y = df['topic']





#x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = .25, random_state = 23)




#
x_reviewText = df['Review Text']
x_reviewTitle = df['Review Title']
#x_reviewText = df['Review Text']
#x_reviewTitle = df['Review Title']





cntvector_reviewText  = CountVectorizer(stop_words='english')
cntvector_reviewText = cntvector_reviewText.fit(x_reviewText)
x_reviewText = cntvector_reviewText.transform(x_reviewText)
x_reviewText = pd.DataFrame(x_reviewText.todense(),columns=cntvector_reviewText.get_feature_names())





cntvector_reviewTitle  = CountVectorizer(stop_words='english')
cntvector_reviewTitle = cntvector_reviewText.fit(x_reviewTitle)
x_reviewTitle = cntvector_reviewText.transform(x_reviewTitle)
x_reviewTitle = pd.DataFrame(x_reviewTitle.todense(),columns=cntvector_reviewTitle.get_feature_names())




x = pd.concat([x_reviewText,x_reviewTitle],axis = 1)


encoder = LabelBinarizer()
encoder = encoder.fit(y)
y = encoder.transform(y) 


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = .25, random_state = 23)
#model.add(keras.layers.GlobalAveragePooling1D())
#model.add(keras.layers.Dense(16, activation=tf.nn.relu))
#model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

vocab_size = 10000
input_layer = x_train.shape[1]
classes = y_train.shape[1]
model = keras.Sequential()
#model.add(keras.layers.Embedding(vocab_size, 16))
#model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(512, input_shape=(input_layer,)))
#model.add(Activation('relu'))
#model.add(Dropout(0.3))
#model.add(Dense(512))
#model.add(Activation('relu'))
#model.add(Dropout(0.3))
#model.add(Dense(classes))
#model.add(Activation('softmax'))
##model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(64, activation=tf.nn.relu))
model.add(keras.layers.Dense(64, activation=tf.nn.relu))
model.add(keras.layers.Dense(21, activation=tf.nn.sigmoid))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])



history = model.fit(x_train,
                    y_train,
                    epochs=10,
                    batch_size=512,
                    validation_data=(x_test, y_test),
                    verbose=1)

results = model.evaluate(x_test, y_test)

print(results)

