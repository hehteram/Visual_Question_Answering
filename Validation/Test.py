# cd "/home/mohammadtaha/Desktop/Deep Learning/DL_Project/Save"
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import nltk
import string
from gensim.models.keyedvectors import KeyedVectors
import pickle
num_class = 1011
batch_size = 1
sentence_len = 10
vec_len = 300
num_hidden = 1024
display_step = 20

X_I = tf.placeholder("float", [batch_size, 7, 7, 512])
X_Q = tf.placeholder("float", [batch_size, sentence_len, vec_len])

cell = rnn.BasicLSTMCell(num_hidden)
cellout, cell_states = tf.nn.dynamic_rnn(cell, X_Q,dtype=tf.float32)

nm_X_I = tf.subtract(X_I,tf.tile(input=tf.reshape(tf.reduce_mean(X_I, axis=3), shape=(batch_size,7,7,1)), multiples=[1,1,1,512]))
X_I_L2 = tf.divide(nm_X_I, (tf.tile(tf.reshape(tf.sqrt(tf.reduce_sum(tf.multiply(nm_X_I,nm_X_I),axis=3)),shape=(batch_size,7,7,1)), multiples=[1,1,1,512])+1e-8))

C1 = tf.concat([X_I_L2, tf.tile(input=tf.reshape(tensor=cellout[:,9,:], shape=(batch_size,1,1,num_hidden)), multiples=[1,7,7,1])], axis=3)

dropout1 = tf.layers.dropout(inputs=C1, rate=0, training=False)

conv1 = tf.layers.conv2d(
      inputs = dropout1,
      filters=512,
      kernel_size=[1, 1],
      padding="same",
      activation=tf.nn.relu,
      bias_initializer = tf.constant_initializer(value=0.1),
      kernel_initializer= tf.random_normal_initializer(stddev=0.1))

dropout2 = tf.layers.dropout(inputs=conv1, rate=0, training=False)

conv2 = tf.layers.conv2d(
      inputs = dropout2,
      filters=2,
      kernel_size=[1, 1],
      padding="same",
      kernel_initializer=tf.random_normal_initializer(stddev=0.1))

r_X_I_L2 = tf.tile(tf.transpose(tf.reshape(tensor=X_I_L2, shape=(batch_size,49,1,512)), perm=[0,3,1,2]), multiples=[1,1,1,2])
attention = tf.tile(input=tf.nn.softmax(logits=tf.reshape(tensor=conv2, shape=(batch_size,1,49,2))), multiples=[1,512,1,1])
avg_w = tf.reshape(tf.reduce_sum(tf.transpose(tf.multiply(r_X_I_L2,attention),perm=[0,1,3,2]),axis=3),shape=(batch_size,num_hidden))

C2 = tf.concat([avg_w,tf.reshape(cellout[:,9,:],shape=[batch_size,num_hidden])], axis=1)

dropout3 = tf.layers.dropout(inputs=C2, rate=0, training=False)
dense1 = tf.layers.dense(inputs=dropout3, units=1024, activation=tf.nn.relu)

dropout4 = tf.layers.dropout(inputs=dense1, rate=0, training=False)
dense2 = tf.layers.dense(inputs=dropout4, units=num_class)

output = tf.nn.softmax(logits=dense2)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess=sess, save_path='./new_Save')
###################################################################################################
base_model = VGG19(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)

img = image.load_img('./Pic.jpg', target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0) #######
x = preprocess_input(x)

feature = model.predict(x)
#############################
word_vectors = KeyedVectors.load_word2vec_format("./Dataset/GoogleNews-vectors-negative300.bin", binary=True, limit=200000)
###################################################################################################
length = 10
p = string.punctuation
q = "Is a hot hirl in the picture?"
print(q)
for c in p:
    q = q.replace(c, "")
splited = nltk.word_tokenize(q)

question = []
for i in splited:
      if (word_vectors.__contains__(i)):
            question.append(word_vectors[i])
if (len(question)<=length):
      s = []
      for i in range(length-len(question)):
            s.append(np.zeros(shape=300))
      question = s + question
else:
      question = question[0:length]

question = tuple(question)
q = []
q.append(question)
q = tuple(q)
###################################################################################################
indices_answers = pickle.load(open("./Result/Indices_Answers", "rb"))
indices_answers[1010] = 'etc.'
o = sess.run(output, feed_dict={X_I: feature, X_Q: q})
o = np.argmax(o, axis=1)
for i in o:
      print (indices_answers[i])
###################################################################################################
