import json
import nltk
import string
import pickle
from gensim.models.keyedvectors import KeyedVectors

length = 10
p = string.punctuation

val_questions = json.load(open('./Dataset/OpenEnded_mscoco_val2014_questions.json', 'r'))
for i in range(len(val_questions["questions"])):
    s = val_questions["questions"][i]["question"].encode()
    for c in p:
        s = s.replace(c, "")
    val_questions["questions"][i]["question"] = s.decode()


word_vectors = KeyedVectors.load_word2vec_format("./Dataset/GoogleNews-vectors-negative300.bin", binary=True, limit=200000)
splited = []
val_all_questions = []
val_num2vec = []
val_num2vec.append(0*word_vectors['you'])
val_word2num = {' '.decode(): 0}
val_num2word = {0: ' '.decode()}
index = 1
for q in val_questions["questions"]:
    splited = nltk.word_tokenize(q["question"].encode())
    sp = []
    for i in splited:
        if (word_vectors.__contains__(i)):
            if(val_word2num.has_key(i.decode())):
                sp.append(val_word2num[i.decode()])
            else:
                val_word2num[i.decode()] = index
                val_num2word[index] = i.decode()
                val_num2vec.append(word_vectors[i])
                sp.append(index)
                index += 1
    if (len(sp)<=length):
        s = []
        for i in range(length-len(sp)):
            s.append(0)
        sp = s + sp
    else:
        sp = sp[0:length]
    val_all_questions.append(sp)

val_dictionary = {}
val_dictionary['word2num'.decode()] = val_word2num
val_dictionary['num2word'.decode()] = val_num2word
with open('./Result/val_dictionary.json', 'w') as f:
    json.dump(val_dictionary, f)

pickle.dump([val_num2vec, val_all_questions] , open( "./Result/Question_Val", "wb"))
