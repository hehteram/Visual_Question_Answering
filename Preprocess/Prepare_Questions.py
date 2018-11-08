import json
import nltk
import string
import pickle
import numpy as np
from gensim.models.keyedvectors import KeyedVectors

length = 10
p = string.punctuation

train_questions = json.load(open('./Dataset/OpenEnded_mscoco_train2014_questions.json', 'r'))
for i in range(len(train_questions["questions"])):
    s = train_questions["questions"][i]["question"].encode()
    for c in p:
        s = s.replace(c, "")
    train_questions["questions"][i]["question"] = s.decode()


word_vectors = KeyedVectors.load_word2vec_format("./Dataset/GoogleNews-vectors-negative300.bin", binary=True, limit=200000)
splited = []
all_questions = []
num2vec = []
num2vec.append(0*word_vectors['you'])
word2num = {' '.decode(): 0}
num2word = {0: ' '.decode()}
index = 1
for q in train_questions["questions"]:
    splited = nltk.word_tokenize(q["question"].encode())
    sp = []
    for i in splited:
        if (word_vectors.__contains__(i)):
            if(word2num.has_key(i.decode())):
                sp.append(word2num[i.decode()])
            else:
                word2num[i.decode()] = index
                num2word[index] = i.decode()
                num2vec.append(word_vectors[i])
                sp.append(index)
                index += 1
    if (len(sp)<=length):
        s = []
        for i in range(length-len(sp)):
            s.append(0)
        sp = s + sp
    else:
        sp = sp[0:length]
    all_questions.append(sp)

train_dictionary = {}
train_dictionary['word2num'.decode()] = word2num
train_dictionary['num2word'.decode()] = num2word
with open('./Result/train_dictionary.json', 'w') as f:
    json.dump(train_dictionary, f)

pickle.dump([num2vec, all_questions] , open( "./Result/Question_Train", "wb"))
