# -*- coding: utf-8 -*-
"""
Created on Fri May 11 12:31:35 2018

@author: paprasad
"""
"""

"""
import nltk
import pandas as pd
import numpy as np

lemmatizer = nltk.stem.WordNetLemmatizer()
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
folder = '/media/pawan/New Volume/Textclassifier/Bug Traiger'
file_name='/Bug_ReportFinal pipe splitted.csv'
path = folder+file_name
#xl = pd.ExcelFile(path)



from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV


from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english", ignore_stopwords=True)

class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])
stemmed_count_vect = StemmedCountVectorizer(stop_words='english')




parameters = {'vect__ngram_range': [(1, 1), (1, 2)],'tfidf__use_idf': (True, False),'clf__alpha': (1e-2, 1e-3),}

def lemmatize_text(text):
    temp=''
    for w in w_tokenizer.tokenize(text):
        temp = temp+' '+lemmatizer.lemmatize(w)
    return temp



def model(data):
    
    numpy_array = data.as_matrix()
    X  = numpy_array[:,2]
    Y  = numpy_array[:,0]
    
    
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=42)
    #MultinomialNB
    MultinomialNB_clf = Pipeline([('vect', stemmed_count_vect),('tfidf', TfidfTransformer()),('clf', MultinomialNB()),])
    MultinomialNB_clf = MultinomialNB_clf.fit(X_train,Y_train)
    predicted = MultinomialNB_clf.predict(X_test)
    print('accuracy of MultinomialNB_clf', np.mean(predicted == Y_test))
    
    #BernoulliNB
#    BernoulliNB_clf = Pipeline([('vect', stemmed_count_vect),('tfidf', TfidfTransformer()),('clf', BernoulliNB()),])
#    BernoulliNB_clf = BernoulliNB_clf.fit(X_train,Y_train)
#    predicted = BernoulliNB_clf.predict(X_test)
#    print('accuracy of BernoulliNB_clf', np.mean(predicted == Y_test))
#    #SVM 
    SGDClassifier_clf = Pipeline([('vect', stemmed_count_vect),('tfidf', TfidfTransformer()),('clf-svm', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42)),])
    SGDClassifier_clf = SGDClassifier_clf.fit(X_train,Y_train)
    predicted  = SGDClassifier_clf.predict(X_test)
    print('accuracy of SGDClassifier_clf', np.mean(predicted == Y_test))
    
    
    LogisticRegression_clf = Pipeline([('vect', stemmed_count_vect),('tfidf', TfidfTransformer()),('clf-svm', LogisticRegression()),])
    LogisticRegression_clf = LogisticRegression_clf.fit(X_train,Y_train)
    predicted  = LogisticRegression_clf.predict(X_test)
    print('accuracy of LogisticRegression_clf', np.mean(predicted == Y_test))
    
    SVC_clf = Pipeline([('vect', stemmed_count_vect),('tfidf', TfidfTransformer()),('clf-svm', SVC()),])
    SVC_clf = SVC_clf.fit(X_train,Y_train)
    predicted  = SVC_clf.predict(X_test)
    print('accuracy of SVC_clf', np.mean(predicted == Y_test))
    
    LinearSVC_clf = Pipeline([('vect', stemmed_count_vect),('tfidf', TfidfTransformer()),('clf-svm', LinearSVC()),])
    LinearSVC_clf = LinearSVC_clf.fit(X_train,Y_train)
    predicted  = LinearSVC_clf.predict(X_test)
    print('accuracy of LinearSVC_clf', np.mean(predicted == Y_test))
    
    
    NuSVC_clf = Pipeline([('vect', stemmed_count_vect),('tfidf', TfidfTransformer()),('clf-svm', NuSVC()),])
    NuSVC_clf = NuSVC_clf.fit(X_train,Y_train)
    predicted  = NuSVC_clf.predict(X_test)
    print('accuracy of NuSVC_clf', np.mean(predicted == Y_test))
#    gs_clf = GridSearchCV(MultinomialNB_clf, parameters, n_jobs=-1)
#    gs_clf = gs_clf.fit(X_train, Y_train)
#    score_1 = gs_clf.best_score_
#    param_1 = gs_clf.best_params_
#    print(param_1)

df = pd.read_csv(path,encoding='iso-8859-1')
df = df.dropna(how='any',axis=0)
df = df.replace(r'\\n',' ', regex=True) 
df.columns = ['Assignee', 'Email']
df['text_lemmatized'] = df.Email.apply(lemmatize_text)
#print (df.iloc[:,2])

#print (df)
model(df)       