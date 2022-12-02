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
nltk.download("unversal_tagset")