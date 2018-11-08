import numpy as np
import pickle
import tensorflow as tf
import h5py
import random
from tensorflow.contrib import rnn

features = h5py.File('Dataset/data_img_pool5.h5', 'r')
features = features['images_train']

answers_index = pickle.load(open( "./Result/Answers_Train", "rb"))
images_index = pickle.load(open( "./Result/Train_Question_Image", "rb"))
num2vec, questions = pickle.load(open( "./Result/Question_Train", "rb"))
tmp = list(zip(answers_index, images_index, questions))
random.shuffle(tmp)
answers_index, images_index, questions = zip(*tmp)

num_class = 1011
batch_size = 128
sentence_len = 10
vec_len = 300
max_num = len(questions)
num_hidden = 1024
display_step = 20

Train = True
Load = False

def newbatch(start_point):
    X_Image = []
    X_Question = []
    Y = []
    for i in range(start_point, start_point+batch_size):
        z = np.zeros(shape=(1011))
        z[answers_index[i]] = 1
        Y.append(z)
        X_Image.append(np.transpose(features[images_index[i]], (1,2,0)))
        X_Question.append(tuple([num2vec[j] for j in questions[i]]))
    return tuple(X_Image), tuple(X_Question), tuple(Y)


X_I = tf.placeholder("float", [batch_size, 7, 7, 512])
X_Q = tf.placeholder("float", [batch_size, sentence_len, vec_len])
Y = tf.placeholder("float", [batch_size, num_class])


cell = rnn.BasicLSTMCell(num_hidden)
cellout, cell_states = tf.nn.dynamic_rnn(cell, X_Q,dtype=tf.float32)

nm_X_I = tf.subtract(X_I,tf.tile(input=tf.reshape(tf.reduce_mean(X_I, axis=3), shape=(batch_size,7,7,1)), multiples=[1,1,1,512]))
X_I_L2 = tf.divide(nm_X_I, (tf.tile(tf.reshape(tf.sqrt(tf.reduce_sum(tf.multiply(nm_X_I,nm_X_I),axis=3)),shape=(128,7,7,1)), multiples=[1,1,1,512])+1e-8))

C1 = tf.concat([X_I_L2, tf.tile(input=tf.reshape(tensor=cellout[:,9,:], shape=(batch_size,1,1,num_hidden)), multiples=[1,7,7,1])], axis=3)

dropout1 = tf.layers.dropout(inputs=C1, rate=0.5, training=True)

conv1 = tf.layers.conv2d(
      inputs = dropout1,
      filters=512,
      kernel_size=[1, 1],
      padding="same",
      activation=tf.nn.relu,
      bias_initializer = tf.constant_initializer(value=0.1),
      kernel_initializer= tf.random_normal_initializer(stddev=0.1))

dropout2 = tf.layers.dropout(inputs=conv1, rate=0.5, training=True)

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

dropout3 = tf.layers.dropout(inputs=C2, rate=0.5, training=True)
dense1 = tf.layers.dense(inputs=dropout3, units=1024, activation=tf.nn.relu)

dropout4 = tf.layers.dropout(inputs=dense1, rate=0.5, training=True)
dense2 = tf.layers.dense(inputs=dropout4, units=num_class)

output = tf.nn.softmax(logits=dense2)

Step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(learning_rate=1e-3, global_step=Step, decay_steps=50000, decay_rate=0.5)
loss = tf.losses.softmax_cross_entropy(onehot_labels=Y, logits=dense2)

tf.summary.scalar("loss", loss)
merge = tf.summary.merge_all()
filewriter = tf.summary.FileWriter('log')

train = tf.train.AdamOptimizer(learning_rate).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

if Train:
    if Load:
        saver.restore(sess=sess, save_path='./new_Save')
    starter = 0
    for iteration in range(0, 60000):
            print iteration
            if ((starter + batch_size) > max_num - 1):
                starter = 0
            i, q, l = newbatch(starter)
            starter = starter + batch_size
            sess.run(train, feed_dict={X_I:i, X_Q:q, Y:l, Step:iteration})
            if iteration % display_step == 0:
                a, b = sess.run((loss, merge), feed_dict={X_I:i, X_Q:q, Y:l, Step:iteration})
                filewriter.add_summary(b, iteration)
                if iteration % 500 == 0:
                    saver.save(sess=sess, save_path='./new_Save')
    saver.save(sess=sess, save_path='./new_Save')
else:
    saver.restore(sess=sess, save_path='./new_Save')
