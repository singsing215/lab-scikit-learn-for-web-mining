# -*- coding: utf-8 -*-
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import numpy as np

categories = ['alt.atheism', 'comp.graphics', 'sci.med', 'soc.religion.christian']
twenty_train = fetch_20newsgroups(subset='train', categories=categories)

# tokenizing text with sk-learn
count_vect = CountVectorizer()#class
X_train_counts = count_vect.fit_transform(twenty_train.data)#)#(doc, token): Occurrence--train

# tf–idf can be computed as follows:
tfidf_transformer = TfidfTransformer()#tf–idf function
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print(type(X_train_tfidf))#class 'scipy.sparse.csr.csr_matrix'
print(X_train_tfidf)#(doc, token): tf–idf--train

# train classifier Naïve Bayes Classifier
clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)
print(clf)#MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)

# Prepare the testing data set
twenty_test = fetch_20newsgroups(subset='test', categories=categories)
X_test_counts = count_vect.transform(twenty_test.data)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)
print(X_test_counts)#(doc, token): Occurrence--test
print(X_test_tfidf)#(doc, token): tf–idf--test

# use the trained classifier to predict results for testing data set
predicted = clf.predict(X_test_tfidf)
print(predicted)#[2 2 3 ... 2 2 1]

#test doc classified as train[category]
for doc, category in zip(twenty_test.data, predicted):
   print('Classified as: %s\n%s\n' % (twenty_train.target_names[category], doc))


print('Accuracy: %.3f\n' % np.mean(predicted == twenty_test.target))#Accuracy: 0.835

# confusion martix
print('Confusion Martix:')
print(metrics.confusion_matrix(twenty_test.target, predicted))
print('\n\n')

# classifaction report: precision, recall, f1-score, support
print('Classification Report:')
print(metrics.classification_report(twenty_test.target, predicted, target_names=twenty_test.target_names))















