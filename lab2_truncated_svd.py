# -*- coding: utf-8 -*-
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression#we use Logisitic Regression11 classifier rather than Naïve Bayes, because the latter cannot handle negative features that are commonly seen after dimension reduction.
from sklearn.decomposition import TruncatedSVD
import numpy as np


categories = ['alt.atheism', 'comp.graphics', 'sci.med', 'soc.religion.christian']
twenty_train = fetch_20newsgroups(subset='train', categories=categories)
twenty_test = fetch_20newsgroups(subset='test', categories=categories)

# raw frequency
count_vect = CountVectorizer()
X_train = count_vect.fit_transform(twenty_train.data)
X_test = count_vect.transform(twenty_test.data)
#print(X_train)#(doc, token): Occurrence--train
#1print(X_test)#(doc, token): Occurrence--test

method = int(input('input your data pre-processing method: 0 (default) for raw \
freqency, 1 for tf-idf, and 2 for tf-idf with truncated svd: '))

if method >= 1:
    # tf-idf
    tfidf_transformer = TfidfTransformer()
    X_train = tfidf_transformer.fit_transform(X_train)
    X_test = tfidf_transformer.transform(X_test)
    
if method == 2:
    svd = TruncatedSVD(n_components=50, n_iter=25, random_state=12)#n_components:dimension of reduced vectors 
    X_train = svd.fit_transform(X_train)
    X_test = svd.transform(X_test)

clf = LogisticRegression().fit(X_train, twenty_train.target)
predicted = clf.predict(X_test)
'''
predicted = clf.predict(X_test_tfidf)
print(predicted)#[2 2 3 ... 2 2 1]
'''
print('Accuracy: %.3f\n' % np.mean(predicted == twenty_test.target))
