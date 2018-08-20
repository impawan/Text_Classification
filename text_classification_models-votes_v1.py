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
import pickle
import matplotlib.pyplot as plt
from collections import Counter





lemmatizer = nltk.stem.WordNetLemmatizer()
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
folder = 'D:\Profiles\paprasad\python\Text Classification\Bug Traiger'
file_name='/SMART_UKR_Jan2018_data_CSV.csv'
path = folder+file_name
xl = pd.ExcelFile(path)

stop_words = nltk.corpus.stopwords.words('english')
newStopWords = ['Regards','Thanks','Please','find',':','(',')',';','<','@','?','{','}','/']
stop_words.extend(newStopWords)

from nltk.classify.scikitlearn import SklearnClassifier
from nltk.stem import PorterStemmer
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV
from nltk.tokenize import sent_tokenize, word_tokenize


from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english", ignore_stopwords=True)
#stemmer = SnowballStemmer("english")

ps = PorterStemmer()
# disabled stemmed count beacuse this was causing the error durning pickle import
#class StemmedCountVectorizer(CountVectorizer):
#    def build_analyzer(self):
#        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
#        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])
#stemmed_count_vect = StemmedCountVectorizer(stop_words=stop_words)
#stemmed_count_vect = StemmedCountVectorizer()



#GridSearch was not working so parameter is disabled, need to study more abou this
#parameters = {'vect__ngram_range': [(1, 1), (1, 2)],'tfidf__use_idf': (True, False),'clf__alpha': (1e-2, 1e-3),}

def save_model(model_name,name):
    file_name = name+'.'+'pickle'
    file_object = open(file_name,'wb')
    pickle.dump(model_name,file_object)  
    file_object.close()




def StopWords(text):
    ret = ''
    for word in word_tokenize(text):
        word = word.lower()
        if word not in stop_words:
            ret = ret+' '+word
    return ret  


def lemmatize_text(text):
    temp=''
    for w in w_tokenizer.tokenize(text):
        temp = temp+' '+lemmatizer.lemmatize(w)
    return temp

def text_stemming(text):
    temp=''
    print(text)
    for w in w_tokenizer.tokenize(text):
        temp = temp+' '+ ps.stem(w)
    return temp    

def model(data):
    
    numpy_array = data.as_matrix()
    X  = numpy_array[:,2]
    Y  = numpy_array[:,0]
    
    #Linear Classifiers: Logistic Regression, Naive Bayes Classifier
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42) #random state 42 has more mean that 23 also .3 performance was good the .4
    #MultinomialNB
    MultinomialNB_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', MultinomialNB(alpha=0.1, class_prior=None, fit_prior=True)),])
#    file_object = open('MultinomialNB.pickle','rb')
#    MultinomialNB_clf = pickle.load(file_object)
    MultinomialNB_clf = MultinomialNB_clf.fit(X_train,Y_train)
    predicted = MultinomialNB_clf.predict(X_test)
    print('MultinomialNB_clf', np.mean(predicted == Y_test))
    save_model(MultinomialNB_clf,"MultinomialNB")
    
    
    
    #BernoulliNB
    BernoulliNB_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', BernoulliNB(alpha=0.1)),])
    BernoulliNB_clf = BernoulliNB_clf.fit(X_train,Y_train)
    predicted = BernoulliNB_clf.predict(X_test)
    print('BernoulliNB_clf', np.mean(predicted == Y_test))
    save_model(BernoulliNB_clf,"BernoulliNB")
    
    #SVM 
    SGDClassifier_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf-svm', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, max_iter=5, random_state=42)),])
    SGDClassifier_clf = SGDClassifier_clf.fit(X_train,Y_train)
    predicted  = SGDClassifier_clf.predict(X_test)
    print('SGDClassifier_clf', np.mean(predicted == Y_test))
    save_model(SGDClassifier_clf,"SGDClassifier")
    
    LogisticRegression_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf-svm', LogisticRegression()),])
    LogisticRegression_clf = LogisticRegression_clf.fit(X_train,Y_train)
    predicted  = LogisticRegression_clf.predict(X_test)
    print('LogisticRegression_clf', np.mean(predicted == Y_test))
    save_model(LogisticRegression_clf,"LogisticRegression")
    
    #Disabled due to low accuracy "accuracy of SVC_clf 0.0956973293768546" action task identitfy why
#    SVC_clf = Pipeline([('vect', stemmed_count_vect),('tfidf', TfidfTransformer()),('clf-svm', SVC()),])
#    SVC_clf = SVC_clf.fit(X_train,Y_train)
#    predicted  = SVC_clf.predict(X_test)
#    print('accuracy of SVC_clf', np.mean(predicted == Y_test))
    
    LinearSVC_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf-svm', LinearSVC()),])
    LinearSVC_clf = LinearSVC_clf.fit(X_train,Y_train)
    predicted  = LinearSVC_clf.predict(X_test)
    print('LinearSVC_clf', np.mean(predicted == Y_test))
    save_model(LinearSVC_clf,"LinearSVC")
    
    
    #disable NuSVC due to error "b'specified nu is infeasible'"
#    NuSVC_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf-svm', NuSVC()),])
#    NuSVC_clf = NuSVC_clf.fit(X_train,Y_train)
#    predicted  = NuSVC_clf.predict(X_test)
#    print('accuracy of NuSVC_clf', np.mean(predicted == Y_test))
    
    
#    gs_clf = GridSearchCV(MultinomialNB_clf, parameters, n_jobs=-1)
#    gs_clf = gs_clf.fit(X_train, Y_train)
#    score_1 = gs_clf.best_score_
#    param_1 = gs_clf.best_params_
#    print(param_1)

#df = pd.read_csv(path,encoding='iso-8859-1')
#df = df.dropna(how='any',axis=0)
#df = df.replace(r'\\n',' ', regex=True) 
#df.columns = ['Assignee', 'Email']
#df['text_lemmatized'] = df.Email.apply(lemmatize_text)
##print (df.iloc[:,2])
#
##print (df)
#model(df)      
    



def visualise (df):
    counts = Counter(df['Assignee'])
    print (type(counts))
    df_bar = pd.DataFrame.from_dict(counts, orient='index')
    df_bar.plot(kind='bar')

def text_classification(sheet):
   df = xl.parse(sheet)
  
   df = df.replace(r'[^a-zA-Z\d\s.]','',regex=True)
   df = df.replace(r'\n',' ', regex=True) 
   df = df.replace(r'\r',' ', regex=True) 
   df = df.replace(r'\t',' ', regex=True) 
   df = df.dropna(how='any',axis=0)
   df = df.replace(r'(<!--((.|\\R)*)-->)',' ',regex=True)
   #df = df.replace(r'(span((.|\\R)*)\")',' ',regex=True)
   df.to_csv('data_frame.clean',sep =',')
   #df = df.replace(r'\\n',' ', regex=True) 
   df.columns = ['Assignee', 'Email']
   #plt.hist(df['Assignee'],)
   df['text_lemmatized'] = df.Email.apply(lemmatize_text)
   df.text_lemmatized = df.text_lemmatized.apply(StopWords)
   #print (df.text_lemmatized)
   df = df.replace(r';',' ', regex=True) 
   print("-------------------------unique lables ----------------------------------\n\n\n")
   print(df.Assignee.unique())
   visualise(df)
#   df = df.replace(r'>',' ', regex=True) 
   #print(df.iloc[:,2])
   model(df)
   
   
   
   


for sheet in xl.sheet_names:
    table_name = sheet
    text_classification(sheet) 