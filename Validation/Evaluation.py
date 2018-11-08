import pickle

val_out = pickle.load(open("./Result/new_Prediction", "rb"))
check_list = pickle.load(open("./Result/Answers_Index_Choice", "rb"))
answers_index = pickle.load(open("./Result/Answers_Val", "rb"))

Score = 0
Total = 0
for i in range(len(val_out)):
  Total += 1
  if (check_list[i] != 1010):
    if (val_out[i] != 1010):
      counter = 0
      for j in range(10):
        if (val_out[i] == answers_index[j][i]):
          counter +=1
      Score += (min(3,counter)/3.0)
  # else:
  #   Total += 1
  #   if (val_out[i] == 1010):
  #     Score += 1
print (Score/Total)
