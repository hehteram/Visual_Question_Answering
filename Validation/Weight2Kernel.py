OLD_CHECKPOINT_FILE = "./Save"
NEW_CHECKPOINT_FILE = "./nSave"

import tensorflow as tf
vars_to_rename = {
    "rnn/basic_lstm_cell/weights": "rnn/basic_lstm_cell/kernel",
    'rnn/basic_lstm_cell/weights/Adam': 'rnn/basic_lstm_cell/kernel/Adam',
    'rnn/basic_lstm_cell/weights/Adam_1': 'rnn/basic_lstm_cell/kernel/Adam_1',
    "rnn/basic_lstm_cell/biases":  "rnn/basic_lstm_cell/bias",
    "rnn/basic_lstm_cell/biases/Adam": "rnn/basic_lstm_cell/bias/Adam",
    "rnn/basic_lstm_cell/biases/Adam_1": "rnn/basic_lstm_cell/bias/Adam_1"
}
new_checkpoint_vars = {}
reader = tf.train.NewCheckpointReader(OLD_CHECKPOINT_FILE)
for old_name in reader.get_variable_to_shape_map():
  if old_name in vars_to_rename:
    new_name = vars_to_rename[old_name]
  else:
    new_name = old_name
  new_checkpoint_vars[new_name] = tf.Variable(reader.get_tensor(old_name))

init = tf.global_variables_initializer()
saver = tf.train.Saver(new_checkpoint_vars)

with tf.Session() as sess:
  sess.run(init)
  saver.save(sess, NEW_CHECKPOINT_FILE)