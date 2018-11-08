from random import *
import json
import pickle

ordering = json.load(open('./Dataset/img_ordering.json', 'r'))

train = ordering['img_train_ordering']
index_train = {}
index = 0
for i in train:
    index_train[int(i.encode()[-16:-4])] = index
    index += 1

train_questions = json.load(open('./Dataset/OpenEnded_mscoco_train2014_questions.json', 'r'))
train_questions = train_questions["questions"]

train_question_image = []
for i in train_questions:
    if (index_train.has_key(i['image_id'])):
        train_question_image.append(index_train[i['image_id']])
    else:
        train_question_image.append(randint(0, 82460))

pickle.dump(train_question_image, open( "./Result/Train_Question_Image", "wb"))
