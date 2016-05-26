# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 20:31:29 2015

@author: hina
"""
print ()

import sys
import glob
import codecs
from nltk.corpus import stopwords
from stemming.porter2 import stem
import random
from textblob.classifiers import NaiveBayesClassifier
from collections import Counter
import datetime
import string
import re

# movie data set
goodpath = "./MoviePosNeg/pos/*.txt"
goodlabel = "POS"
badpath = "./MoviePosNeg/neg/*.txt"
badlabel = "NEG"

# health data set
#goodpath = "./HealthProNonPro/Pro/*.txt"
#goodlabel = "PRO"
#badpath = "./HealthProNonPro/NonPro/*.txt"
#badlabel = "NONPRO"

#####################################################################
# preprocess text:
#   - remove punctuation 
#   - remove digits
#   - lower case words
#   - remove short words
#   - remove stop words
#   - stem the words
#   - retain only unique words
#
# note: can make this more efficient by combining for loops but leaving separate for clarity
#
def preprocessText (text, rmpunc='y', rmdigits='y', lowerwords='y', minwordlen=1, rmstopwords='y', stemwords='y', uniquewords='n'):
        
    # remove punctuation
    if (rmpunc=='y'):
        regex = re.compile('[%s]' % re.escape(string.punctuation))
        text = regex.sub('', text)
        
    # remove digits
    if(rmdigits=='y'):
        regex = re.compile('[%s]' % re.escape(string.digits))
        text = regex.sub('', text)

    # lower case words
    if (lowerwords=='y'):
        text = text.lower()
        
    # remove short words
    if (minwordlen>1):
        text = ' '.join([word for word in text.split() if (len(word)>=minwordlen)])
        
    # remove stop words
    if (rmstopwords=='y'):
        text = ' '.join([word for word in text.split() if word not in stopwords.words("english")])
        
    # stem the words
    if (stemwords=='y'):
        text = ' '.join([stem(word) for word in text.split()])
            
    # retain only unique words
    if (uniquewords=='y'):
        text = ' '.join(set(text.split()))
        
    # return preprocessed data
    return text    

#####################################################################
# preprocess data - this just loops through the dataset and calls preprocessText for the text in the dataset
def preprocessData (data, rmpunc='y', rmdigits='y', lowerwords='y', minwordlen=1, rmstopwords='y', stemwords='y', uniquewords='n'):
    
    fdata = []

    # note: we can combine a lot of the preprocessing tasks for efficiency,
    # but we're keeping them separate for clarity
    for (text, label) in data:
        
        # prerocess test
        text = preprocessText(text, rmpunc, rmdigits, lowerwords, minwordlen, rmstopwords, stemwords, uniquewords)
        
        # pick up the preprocessed text
        fdata.append((text , label))
    
    # return preprocessed data
    return fdata        
     
#####################################################################
# read in text from path, create (text, label) tuples and append to data[]
def getData (path, label, data, fraction): 
   
    # read data
    files = glob.glob(path)
    for file in files:
        # use Unicode text encoding and ignore any errors 
        with codecs.open(file, "r", encoding='utf-8', errors='ignore') as f:
            text = f.read()
            text = text.replace('\n', ' ')
            data.append((text , label)) 
            
    # shuffle data
    random.shuffle(data)
    
    # pick only fraction of data
    data = data[:int(len(data)*fraction)]   
    
    return data
    
#####################################################################
# partition data into training and test sets
def partitionData (gooddata, baddata, traindata, testdata, partition):    
    
    # partition data into training and testing sets
    traindata = gooddata[:(int(len(gooddata)*partition))] + baddata[:(int(len(baddata)*partition))]
    testdata = gooddata[(int(len(gooddata)*partition)):] + baddata[(int(len(baddata)*partition)):]
    
    # return training and test sets
    return (traindata, testdata)

#####################################################################
print ("START:", datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'))
print ()

# read in good and bad datasets as list of (text, label) tuples
# run analysis on only a fraction of the dataset if desired  
print ("Reading datasets...")
print ()
sys.stdout.flush()
gooddata = []
baddata = []
gooddata = getData(goodpath, goodlabel, gooddata, fraction=1/4)
baddata = getData(badpath, badlabel, baddata, fraction=1/4)

# partition good and bad datasets into training and testing sets
print ("Partitioning datasets...")
print ()
sys.stdout.flush()
traindata = []
testdata = []
traindata, testdata = partitionData(gooddata, baddata, traindata, testdata, partition=2/3)

# preproccess training and testing datasets
# without preprocessing, classification will typically take longer and have lower accuracy  
print ("Preprocessing datasets...")
print ()
sys.stdout.flush()
traindata = preprocessData(traindata, minwordlen=4)
testdata = preprocessData(testdata, minwordlen=4)

# train the Naive Bayes Classifier
print ("Training Naive Bayes Classifier...")
print ()
sys.stdout.flush()
nbc = NaiveBayesClassifier(traindata)

# show the most informative features used for classification
nbc.show_informative_features(5)
print ()
sys.stdout.flush()

# test the Naive Bayes Classifier
print ("Testing Naive Bayes Classifier...")
sys.stdout.flush()
acc = nbc.accuracy(testdata)
print ("Accuracy:", round (acc, 4))
print ()

# print the confusion matrix
print ("Printing Confusion Matrix...")
print ()
sys.stdout.flush()

conf = []
for row in testdata:
	conf.append((row[1], nbc.classify(row[0])))	
Counter(conf)

print ("Total size of test data :	%d" %len(testdata))
print ("Original(>) Predicted (V)") 
print ("	", goodlabel, "			", badlabel)
print (goodlabel, "	 %d				%d"  %(Counter(conf)[goodlabel, goodlabel], Counter(conf)[goodlabel, badlabel]))
print (badlabel, "	 %d				%d"  %(Counter(conf)[badlabel, goodlabel], Counter(conf)[badlabel, badlabel]))
print ()

print ("END:", datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'))
print ()




