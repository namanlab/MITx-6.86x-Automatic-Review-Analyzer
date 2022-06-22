from string import punctuation, digits
import numpy as np
import random
import traceback
import project1 as p1

import utils
train_data = utils.load_data('reviews_train.tsv')
val_data = utils.load_data('reviews_val.tsv')
test_data = utils.load_data('reviews_test.tsv')
train_texts, train_labels = zip(*((sample['text'], sample['sentiment']) for sample in train_data))
val_texts, val_labels = zip(*((sample['text'], sample['sentiment']) for sample in val_data))
test_texts, test_labels = zip(*((sample['text'], sample['sentiment']) for sample in test_data))

f = open("stopwords.txt", "r")
data = f.read()
data_into_list = data.replace('\n', ' ').split(" ")
f.close()
print(data_into_list)

for i in train_texts:
    print(i)


