# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 09:49:51 2018

@author: paprasad
"""

import numpy as np
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.pipeline import Pipeline

path = "D:/Profiles/paprasad/Downloads/20news-bydate-train"
print (path)

data_train =load_files(path,description=None, categories=None, load_content=True, shuffle=True, encoding='ISO-8859-1', decode_error='strict', random_state=0)



count_vect = CountVectorizer()

X_train_counts = count_vect.fit_transform(data_train.data)

#print (X_train_counts)



tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
#print (X_train_tfidf.shape)

#Naive Base classifier
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, data_train.target)






text_clf = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', MultinomialNB()),
 ])


text_clf = text_clf.fit(data_train.data, data_train.target)

path_test = "D:/Profiles/paprasad/Downloads/20news-bydate-test"


data_test =load_files(path_test,description=None, categories=None, load_content=True, shuffle=True, encoding='ISO-8859-1', decode_error='strict', random_state=0)

predicted = text_clf.predict(data_test.data)
np.mean(predicted == data_test.target)


#using SVM


from sklearn.linear_model import SGDClassifier


text_clf_svm = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',
                                            alpha=1e-3, n_iter=5, random_state=42)),
 ])


text_clf_svm  = text_clf_svm.fit(data_train.data, data_train.target)
predicted_svm = text_clf_svm.predict(data_test.data)
np.mean(predicted_svm == data_test.target)