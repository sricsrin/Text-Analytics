# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 18:08:28 2015

@author: hina
"""

# TFIDF code adapted from: 
# http://stevenloria.com/finding-important-words-in-a-document-using-tf-idf/

import math
from textblob import TextBlob 
import string
import re
from nltk.corpus import stopwords
from stemming.porter2 import stem

print ()

# "Python" definitions from Wikipedia

document1 = """In Greek mythology, Python (Greek: Πύθων, gen.: Πύθωνος) was the earth-dragon of 
Delphi, always represented in Greek sculpture and vase-paintings as a serpent. He presided at the 
Delphic oracle, which existed in the cult center for his mother, Gaia, "Earth," Pytho being the 
place name that was substituted for the earlier Krisa.[1] Hellenes considered the site to be the 
center of the earth, represented by a stone, the omphalos or navel, which Python guarded."""

document2 = """Monty Python (sometimes known as The Pythons)[2][3] were a British surreal comedy 
group who created the sketch comedy show Monty Python's Flying Circus, that first aired on the BBC on 
October 5, 1969. Forty-five episodes were made over four series. The Python phenomenon developed from 
the television series into something larger in scope and impact, spawning touring stage shows, films, 
numerous albums, several books, and a stage musical. The group's influence on comedy has been compared 
to The Beatles' influence on music."""

document3 = """Python is a widely used general-purpose, high-level programming language.[19][20] 
Its design philosophy emphasizes code readability, and its syntax allows programmers to express 
concepts in fewer lines of code than would be possible in languages such as C++ or Java.[21][22] 
The language provides constructs intended to enable clear programs on both a small and large scale."""

#####################################################################
# preprocess dataset:
#   - remove punctuation 
#   - remove digits
#   - lower case words
#   - remove short words
#   - remove stop words
#   - stem the words
#   - retain only unique words
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

# preprocess all the documents first
print("Preprocessed docs...\n")  
document1 = preprocessText(document1, stemwords='n')
print("Document 1")
print(document1,"\n")
document2 = preprocessText(document2, stemwords='n')
print("Document 2")
print(document2,"\n")
document3 = preprocessText(document3, stemwords='n')
print("Document 3")
print(document3,"\n")

# create text blobs from documents
blob1 = TextBlob(document1)
blob2 = TextBlob(document2)
blob3 = TextBlob(document3)

# create bloblist
bloblist = [blob1, blob2, blob3]

# get tfidf
for i, blob in enumerate(bloblist):
    
    print ("\nTop TF-IDFs for Document", i+1)
    
    tfidf = {}
    
    for term in blob.words:
    
        # Term Frequency (TF) in doc:
        #     Number of times a term appears in a document, 
        #     normalized by dividing by the total number of terms in the document
        tf = blob.words.count(term) / len(blob.words)
        
        # number of documents
        nd = len(bloblist)

        # number of documents with term
        ndwt = sum(1 for b in bloblist if term in b)
        
        # Inverse Document Frequency (IDF)
        #     Measures how common a term is among all documents. 
        #     The more common a term is, the lower its IDF. 
        #     Take the log of the ratio of the total number of documents to 
        #     the number of documents containing the term. 
        #     Add 1 to the divisor to prevent division by zero.
        idf = math.log(nd/(1+ndwt))
        
        # TFIDF = TF*IDF
        # Intuitively:
        # If a term appears frequently in a document, it's important - give the term a high score.
        # But if a term appears in many documents, it's not a unique identifier - give the term a low score.
        tfidf[term] = round(tf*idf, 5)
    
    # sort terms in document by TFIDF
    tfidf = sorted(tfidf.items(), key=lambda x: x[1], reverse=True)
    print (tfidf[:3])
