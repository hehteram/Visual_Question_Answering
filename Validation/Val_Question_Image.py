import json
import pickle

ordering = json.load(open('./Dataset/img_ordering.json', 'r'))

val = ordering['img_val_ordering']
index_val = {}
index = 0
for i in val:
    index_val[int(i.encode()[-16:-4])] = index
    index += 1

val_questions = json.load(open('./Dataset/OpenEnded_mscoco_val2014_questions.json', 'r'))
val_questions = val_questions["questions"]

val_question_image = []
for i in val_questions:
    if (index_val.has_key(i['image_id'])):
        val_question_image.append(index_val[i['image_id']])

pickle.dump(val_question_image, open( "./Result/Val_Question_Image", "wb"))
