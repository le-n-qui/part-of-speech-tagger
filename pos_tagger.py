# Part of Speech Tagger

# import libraries
import nltk
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import pprint, time

# download the Brown corpus
from nltk.corpus import brown

# download the universal tagset from nltk
nltk.download("universal_tagset")

# retrieve word and its tag in brown corpus
data = list(brown.tagged_sents(tagset="universal"))

# create train and test set
train, test = train_test_split(data, train_size=0.8, test_size=0.2, random_state=42)

# create train tagged words
train_tagged_words = [pair for sent in train for pair in sent]
# create test tagged words
test_tagged_words = [pair for sent in test for pair in sent]
