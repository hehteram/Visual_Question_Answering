import json
import pickle

val_answers = json.load(open('./Dataset/mscoco_val2014_annotations.json', 'r'))["annotations"]
siz = 1010
answers_indices = pickle.load(open("./Result/Answers_Indices", "rb"))
###########################################################################

answers_index_choice = []
for annotation in val_answers:
    if (answers_indices.has_key(annotation["multiple_choice_answer"].encode()) == True):
        i = answers_indices[annotation["multiple_choice_answer"].encode()]
    else:
        i = siz
    answers_index_choice.append(i)
pickle.dump(answers_index_choice, open( "./Result/Answers_Index_Choice", "wb"))


counter = 0
answers_val_list = []
length = len(val_answers[0]['answers'])
for j in range(length):
    temp = []
    for annotation in val_answers:
        if (answers_indices.has_key(annotation['answers'][j]['answer']) == True):
            i = answers_indices[annotation['answers'][j]['answer']]
            counter += 1
        else:
            i = siz
        temp.append(i)
    answers_val_list.append(temp)
pickle.dump(answers_val_list, open( "./Result/Answers_Val", "wb"))