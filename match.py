import tensorflow as tf
import numpy as np

def cosine_similarity(a, b):
	normalize_a = tf.nn.l2_normalize(a,0)        
	normalize_b = tf.nn.l2_normalize(b,0)
	cos_similarity=tf.reduce_sum(tf.multiply(normalize_a,normalize_b))
	return cos_similarity

def fm(v1, v2, W):
	m[i] = cosine_similarity(tf.matmul(W[i],v1), tf.matmul(W[i],v2))
	pass

B = 8
lp = 82
lh = 62
h = 128

P_fw = tf.Variable(np.random.rand(B, lp, h))
P_fw_i = P_fw[0][0][0]
P_bw = tf.Variable(np.random.rand(B, lp, h))
H_fw = tf.Variable(np.random.rand(B, lh, h))
H_bw = tf.Variable(np.random.rand(B, lh, h))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)
    _, _, _, _, p_fw_i = sess.run([P_fw, P_bw, H_fw, H_bw, P_fw_i])
    print(p_fw_i)


