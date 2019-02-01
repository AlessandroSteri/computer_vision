import tensorflow as tf
import numpy as np

B = 8
lp = 82
lh = 62
h = 128

P_fw = tf.Variable(np.random.rand(B, lp, h))
P_bw = tf.Variable(np.random.rand(B, lp, h))
H_fw = tf.Variable(np.random.rand(B, lh, h))
H_bw = tf.Variable(np.random.rand(B, lh, h))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)
    p_fw = sess.run([P_fw, P_bw, H_fw, H_bw])


