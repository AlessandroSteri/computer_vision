import tensorflow as tf

def cnn_auto_encoder(data, batch_size, dim_width=256, dim_height=256, nchannels=3):
    # data = tf.placeholder(tf.float32, [batch_size, dim_width, dim_height, nchannels], name='raw_img')
    # TODO use range

    # ENCODE
    # import ipdb; ipdb.set_trace()  # TODO BREAKPOINT
    enc_conv0 = tf.layers.conv2d(data, 64, 3, activation=tf.nn.relu)
    enc_conv0 = tf.layers.max_pooling2d(enc_conv0, 2, 2)
    enc_conv1 = tf.layers.conv2d(enc_conv0, 32, 3, activation=tf.nn.relu)
    enc_conv1 = tf.layers.max_pooling2d(enc_conv1, 2, 2)
    enc_conv2 = tf.layers.conv2d(enc_conv1, 16, 3, activation=tf.nn.relu)
    enc_conv2 = tf.layers.max_pooling2d(enc_conv2, 2, 2)  # TODO check shape and return

    # DECODE
    dec_conv2 = tf.image.resize_nearest_neighbor(enc_conv2, tf.constant([64, 64]))  # q/4
    dec_conv2 = tf.layers.conv2d(dec_conv2, 32, 3, activation=tf.nn.relu)
    dec_conv1 = tf.image.resize_nearest_neighbor(dec_conv2, tf.constant([128, 128]))  # q/2
    dec_conv1 = tf.layers.conv2d(dec_conv1, 64, 3, activation=tf.nn.relu)
    dec_conv0 = tf.image.resize_nearest_neighbor(dec_conv1, tf.constant([256, 256]))  # q
    logits =    tf.layers.conv2d(dec_conv0, 3, (3, 3), padding='same', activation=None,name="logits")
    loss = tf.pow(data - logits, 2)
    cost = tf.reduce_mean(loss)
    # opt = tf.train.AdamOptimizer(0.001).minimize(cost)
            # img,batch_cost, _ = sess.run([logits,cost, opt ], feed_dict={data: batch_data,targets_data: batch_fin})
        # ev=img[0]
        # ev=ev.astype(int)
        # ev[np.where(ev<0)]=0
        # cv.imwrite('./restmp/res-'+str(e_vals) + '.jpg', ev)
#### Inference Code
        # img = sess.run([logits], feed_dict={data: np.array([infer_data[i]])})
    return cost, enc_conv2

def cnn_img_encoder(data, batch_size, dim_width=256, dim_height=256, nchannels=3):
    # data = tf.placeholder(tf.float32, [batch_size, dim_width, dim_height, nchannels], name='raw_img')
    # TODO use range

    # ENCODE
    enc_conv0 = tf.layers.conv2d(data, 64, 3, activation=tf.nn.relu)
    enc_conv0 = tf.layers.max_pooling2d(enc_conv0, 2, 2)
    enc_conv1 = tf.layers.conv2d(enc_conv0, 32, 3, activation=tf.nn.relu)
    enc_conv1 = tf.layers.max_pooling2d(enc_conv1, 2, 2)
    enc_conv2 = tf.layers.conv2d(enc_conv1, 16, 3, activation=tf.nn.relu)
    enc_conv2 = tf.layers.max_pooling2d(enc_conv2, 2, 2)  # TODO check shape and return
    return enc_conv2
