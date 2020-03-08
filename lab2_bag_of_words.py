# -*- coding: utf-8 -*-

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

categories = ['alt.atheism', 'comp.graphics', 'sci.med', 'soc.religion.christian']
twenty_train = fetch_20newsgroups(subset='train', categories=categories)

#build a simple bag-of-words feature vector
count_vect = CountVectorizer()#class CountVectorizer()
bow_train_counts = count_vect.fit_transform(twenty_train.data)#function fit_transform()

print('Number of documents in twenty_train.data:%s' % len(twenty_train.data))#2257 documents

print('Number of extracted features:%s' %len(count_vect.get_feature_names()))#35788 tokens

print('Size of bag-of-words:')
print(bow_train_counts.shape)#2,257 documents and 35,788 tokens (distinct terms)
print('')
print('Bag of words: [(doc_id, features_id): Occurrence]')#(doc, token): Occurrence
print(bow_train_counts)#bag of words feature vector
print('')
print(bow_train_counts[0])#doc_0
print(bow_train_counts[0, 32493])#times
print(count_vect.get_feature_names())#token name


