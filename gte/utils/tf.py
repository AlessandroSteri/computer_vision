import math
import tensorflow as tf
from tensorflow.python.keras import backend as K
from as_srl.utils.lmd import identity

# in_size -> out_size
def fully_connected_layer(x, in_size:int, out_size:int, activation=identity, name:str=''):
    if name != '': name = '_' + name
    W = tf.Variable(tf.truncated_normal([in_size, out_size], stddev=0.1 / math.sqrt(out_size)), name='weight' + name)
    b = tf.Variable(tf.constant(0.1, shape=[out_size]), name='bias'+name)
    y = tf.nn.xw_plus_b(x, W, b, name=name)
    return activation(y)

def bilstm_layer(x, sequence_length, hidden_size, name=''):
    if name != '': name = '_' + name
    with tf.name_scope('bilstm_' + str(name)):
        cell_fw = tf.contrib.rnn.LSTMCell(hidden_size, name='cell_fw' + str(name))
        cell_bw = tf.contrib.rnn.LSTMCell(hidden_size, name='cell_bw' + str(name))
        (out_fw, out_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, x, sequence_length=sequence_length, dtype=tf.float32)
        return tf.concat([out_fw, out_bw], axis=-1, name='latent_repr' + str(name))

# size -> size
def highway(x, size:int, activation=identity, carry_b:int=-1.0, name:str=''):
    if name != '': name = '_' + name

    # flattens the input
    x_shape = tf.shape(x)
    x = tf.reshape(x, (-1, size))
    W = tf.Variable(tf.truncated_normal([size, size], stddev=0.1 / math.sqrt(size)), name='weight' + name)
    b = tf.Variable(tf.constant(0.1, shape=[size]), name='bias' + name)

    W_T = tf.Variable(tf.truncated_normal([size, size], stddev=0.1 / math.sqrt(size)), name='weight_transform' + name)
    b_T = tf.Variable(tf.constant(carry_b, shape=[size]), name='bias_transform' + name)

    activation_op = activation(tf.nn.xw_plus_b(x, W, b, name='transform_gate' + name))
    transform_gate_op = tf.sigmoid(tf.nn.xw_plus_b(x, W_T, b_T, name='transform_gate' + name))
    carry_gate_op = tf.subtract(1.0, transform_gate_op, name='carry_gate' + name)

    gated_activation = tf.multiply(transform_gate_op, activation_op)
    gated_x = tf.multiply(x, carry_gate_op)

    y = tf.add(gated_activation, gated_x)

    # un-flattens the output
    y = tf.reshape(y, x_shape)
    return y

def attention_layer(x, sequence_length, batch_size:int, time_size:int, hidden_size:int, name:str=''):
    mask = tf.sequence_mask(sequence_length, time_size, name='attention_mask')
    mask = tf.reshape(mask, (batch_size, time_size))
    mask = tf.cast(mask, tf.float32)
    W = tf.Variable(tf.random_uniform([hidden_size, 1], minval=-1, maxval=1))
    b = tf.Variable(tf.zeros(shape=(time_size,)))

    u = tf.squeeze(tf.tensordot(x, W, axes=1), -1) # shape (batch_size, time_size) was (batch_size, time_size, 1) before squeeze
    u = u + b # shape (batch_size, time_size)
    u = tf.tanh(u)
    # softmax
    a = tf.exp(u) # shape (batch_size, time_size)
    a = mask * a
    a /= tf.reduce_sum(a, axis=1, keepdims=True) + K.epsilon() # [batch_size, time_size] / [batch_size, 1] due to keepdims
    weighted_x = x * tf.expand_dims(a, -1) # [batch_size, time_size, hidden_size] * [batch_size, time_size, 1]
    y = tf.reduce_sum(weighted_x, axis=1) #[batch_size, hidden_size]
    y = tf.tile(y,[1, time_size]) #SHAPE [B,T*H]
    y = tf.reshape(y, (batch_size, time_size, hidden_size)) #SHAPE [B,T,H]
    return y, a

#TODO
# def character_embedding(batch_size, max_sentence_len, embedding_size, char_dic):
    # pass
