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


def count_word_and_tag(word, tag, train_data):
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

def count_tag2_given_tag1(t2, t1, train_data):
    """This method returns a tuple
       containing the count of tag 2
       given tag 1 and the count of 
       tag 1.
    """
    tags = [pair[1] for pair in train_data]
    t1_count = len([tag for tag in tags if tag == t1])
    count_t2_given_t1 = 0

    for pos in range(len(tags)-1):
        if tags[pos] == t1 and tags[pos+1] == t2:
            count_t2_given_t1 += 1

    return (count_t2_given_t1, t1_count)

def main():
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

    # set of tags
    tags = {tag for word, tag in train_tagged_words}
    # set of words (vocabulary)
    vocab = {word for word, tag in train_tagged_words}

    # create a transition matrix of tags
    # matrix(i, j) represents the Probability(jth tag after ith tag)
    tags_matrix = np.zeros((len(tags), len(tags)), dtype="float32")
    for i, t1 in enumerate(list(tags)):
        for j, t2 in enumerate(list(tags)):
            t2_given_t1_count, t1_count = count_tag2_given_tag1(t2, t1, train_tagged_words)
            tags_matrix[i,j] = t2_given_t1_count/t1_count

    # convert matrix into dataframe
    tags_df = pd.DataFrame(tags_matrix, columns=list(tags), index=list(tags))


if __name__ == "__main__":
    main()