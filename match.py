import tensorflow as tf
import numpy as np
# from match.match_utils import bilateral_match_func

def cosine_similarity(a, b):
    normalize_a = tf.nn.l2_normalize(a, 0)
    normalize_b = tf.nn.l2_normalize(b, 0)
    cos_similarity = tf.reduce_sum(tf.multiply(normalize_a, normalize_b))
    return cos_similarity

# def fm(v1, v2, W):
# 	m[i] = cosine_similarity(tf.matmul(W[i],v1), tf.matmul(W[i],v2))
# 	pass

B = 8
lp = 82
lh = 62
h = 128

P_fw = tf.Variable(np.random.rand(B, lp, h))
P_fw_i = P_fw[0][0][0]
P_bw = tf.Variable(np.random.rand(B, lp, h))
H_fw = tf.Variable(np.random.rand(B, lh, h))
H_bw = tf.Variable(np.random.rand(B, lh, h))
i = 0
def cond(i, size):
    return i < size

def set_zero(i, size):
    i += 1
    # T[i] = 0
    return [i, 0]

T = tf.Variable(np.random.rand(B))

loop = tf.while_loop(cond, set_zero, [i, 5])

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)
    import ipdb; ipdb.set_trace()  # TODO BREAKPOINT
    ll = sess.run([loop])
    print(ll)
