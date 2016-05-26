# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 09:39:12 2015

@author: hina
"""

print ()

from nltk.corpus import stopwords
from stemming.porter2 import stem
import string
import re

# review from Amazon.com
text = """Pirates of The Caribbean is quite simply Hollywood's best pirate 
film in ages; a funny, rollicking swashbuckler that pays homage to the great 
films of the 1930's and 1940's featuring the likes of Errol Flynn, Charles 
Laughton, among others."""

print ("Original:")
print (text)
print ()

# lower case all words
text = text.lower()
print ("Lower Case:")
print (text)
print ()

# remove all punctuation
regex = re.compile('[%s]' % re.escape(string.punctuation))
text = regex.sub('', text)
print ("No Punctuation:")
print (">>>> ", string.punctuation)
print (text)
print ()

# remove all digits
regex = re.compile('[%s]' % re.escape(string.digits))
text = regex.sub('', text)
print ("No Digits:")
print (">>>> ", string.digits)
print (text)
print ()

# only retain words >= 4 characters long
text = ' '.join([word for word in text.split() if (len(word)>=4)])
print ("Only Words >= 4 Chars Long:")
print (text)
print ()

# remove stop words
sw = stopwords.words("english")
text = ' '.join([word for word in text.split() if word not in sw])
print ("No Stop Words:")
print (sw)
print (text)
print ()

# convert to stem words
text = ' '.join([stem(word) for word in text.split()])
print ("Stem Words:")
print (text)
print ()

# only retain unique words
text = ' '.join(set(text.split()))
print ("Unique Words:")
print (text)
print ()
