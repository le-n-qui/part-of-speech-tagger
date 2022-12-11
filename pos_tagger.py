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

def count_word_and_tag(word, tag, train_data=train_tagged_words):
    """This method returns a tuple
       containing the count of the provided
       word with the given tag and the
       count of the provided tag 
       occurring in the train data.
    """
    # save all pairs where the tag of interest occurs
    tag_list = [pair for pair in train_data if pair[1] == tag]

    # count the number of times the tag of interest occurs
    tag_count = len(tag_list)

    # given the tag list, find the words that matche the word of interest
    word_list = [pair[0] for pair in tag_list if pair[0] == word]

    # count the number of times the passed-in word occurs as the passed-in tag
    word_as_tag_count = len(word_list)

    return (word_as_tag_count, tag_count)