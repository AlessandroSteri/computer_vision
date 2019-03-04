import tensorflow as tf
from gte.info import NUM_CLASSES, FEAT_SIZE


# ### OPTIMIZERS ###
def apply_gradient_to_adam(loss, learning_rate, clipper=50, global_step=None):
    # clipper = 50
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    tvars = tf.trainable_variables()
    l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tvars if v.get_shape().ndims > 1])
    new_loss = loss + 1e-5 * l2_loss
    grads, _ = tf.clip_by_global_norm(tf.gradients(new_loss, tvars), clipper)
    optimizer = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)
    return optimizer, new_loss


# ### LOSSES ###
def mean_softax_cross_entropy(logits, labels):
    gold_matrix = tf.one_hot(labels, NUM_CLASSES, dtype=tf.float32)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=gold_matrix))
    return loss


#TODO Agostina, prima di usarlo ricontrolla che non ho fatto danni :)
def image_top_down_attention_layer(I, H_lookup, lengths_H, options, cell_size=256, P_lookup=None, N=512):
    cell_fw_H = tf.contrib.rnn.GRUBlockCellV2(cell_size, name="fw_cells_H")
    # cell_bw_H = tf.contrib.rnn.GRUBlockCellV2(cell_size, name="bw_cells_H")

    outputs, H_states = tf.nn.dynamic_rnn(cell_fw_H, H_lookup, dtype=tf.float32, swap_memory=True, sequence_length=lengths_H)
    # outputs, H_states = tf.nn.bidirectional_dynamic_rnn(cell_fw_H, cell_bw_H, H_lookup, dtype=tf.float32, swap_memory=True, sequence_length=lengths_H)

    if options.with_P_top_down:
        assert P_lookup is not None
        cell_fw_P = tf.contrib.rnn.GRUBlockCellV2(cell_size, name="fw_cells_P")
        cell_bw_P = tf.contrib.rnn.GRUBlockCellV2(cell_size, name="bw_cells_P")
        outputs, P_states = tf.nn.bidirectional_dynamic_rnn(cell_fw_P, cell_bw_P, P_lookup, dtype=tf.float32, swap_memory=True)

        P_embedding = tf.concat([P_states[0], P_states[1]], 1)  # [batch_size x HH]
    else:
        assert P_lookup is None

    # For each location i = 1...K in the image, the feature
    # vector v_i is concatenated with the question embedding q

    # H_embedding = tf.concat([H_states[0], H_states[1]], 1) #[batch_size x HH] GRU
    # H_embedding = tf.concat([H_states[0][0], H_states[0][1]], 1) #[batch_size x HH] biLSTM
    H_embedding = H_states  # [batch_size x H] GRU one direction

    feat_H = tf.map_fn(lambda i: tf.map_fn(lambda x: tf.concat([x, H_embedding[i]], 0), I[i]), tf.convert_to_tensor(list(range(options.batch_size)), dtype=tf.int32), dtype=tf.float32)  # [batch_size x NUM_FEATS x (FEAT_SIZE + HH)]
    # Passed through a non-linear layer f_a...
    HH = options.hidden_size  # * 2
    W_nl = tf.get_variable("W_nl", [N, FEAT_SIZE + HH], dtype=tf.float32)
    b_nl = tf.get_variable("b_nl", [N], dtype=tf.float32)
    feat_H = tf.reshape(tf.transpose(tf.to_float(feat_H), perm=[2,0,1]), [FEAT_SIZE + HH, -1]) #[FEAT_SIZE + HH, batch_size * NUM_FEATS]
    y_att = tf.tanh(tf.transpose(tf.matmul(W_nl, feat_H)) + b_nl) # [batch_size x NUM_FEATS, N]
    W_nl2 = tf.get_variable("W_nl2", [N, FEAT_SIZE + HH], dtype=tf.float32)
    b_nl2 = tf.get_variable("b_nl2", [N], dtype=tf.float32)
    g_att = tf.nn.sigmoid(tf.transpose(tf.matmul(W_nl2, feat_H)) + b_nl2) # [batch_size x NUM_FEATS, N]
    y = tf.multiply(y_att, g_att)  # [batch_size x NUM_FEATS, N]
    W_a = tf.get_variable("W_a", [1, FEAT_SIZE], dtype=tf.float32)
    a = tf.matmul(W_a, tf.transpose(y)) # [1, batch_size x NUM_FEATS]
    #...to obtain a scalar attention weight α_{i,t}
    alpha = tf.nn.softmax(tf.transpose(a)) # [1, batch_size x NUM_FEATS]
    alpha = tf.reshape(alpha, [options.batch_size, NUM_FEATS, 1])
    #The image features are then weighted by the normalized values and summed
    sum_weights = tf.reduce_sum(alpha, axis=1) #[batch_size]
    features_att = tf.map_fn(lambda x: tf.multiply(x[1], x[0]), (I, alpha), dtype=alpha.dtype) #[batch_size x NUM_FEATS x FEAT_SIZE]
    features_att = tf.map_fn(lambda x: tf.reduce_sum(x[0], 0) / x[1], (features_att, sum_weights), dtype=features_att.dtype) #[batch_size x FEAT_SIZE]


    P_dim = options.hidden_size
    if options.with_P_top_down:
        #cell_fw_P = tf.contrib.rnn.GRUBlockCellV2(cell_size, name="fw_cells_P_gru")
        cell_fw_P = tf.contrib.rnn.LSTMCell(cell_size, name="fw_cells_P_lstm")
        #cell_bw_P = tf.contrib.rnn.LSTMCell(cell_size, name="bw_cells_P_lstm")
        outputs, P_states = tf.nn.dynamic_rnn(cell_fw_P, P_lookup, dtype=tf.float32, swap_memory=True, sequence_length=lengths_P)
        output = outputs
        #outputs, P_states = tf.nn.bidirectional_dynamic_rnn(cell_fw_P, cell_bw_P, P_lookup, dtype=tf.float32, swap_memory=True, sequence_length=lengths_P)
        #output = tf.concat(outputs, 2)

        timesteps_H = tf.map_fn(lambda i: tf.map_fn(lambda x: tf.concat([x, H_embedding[i]], 0), output[i]), tf.convert_to_tensor(list(range(options.batch_size)), dtype=tf.int32), dtype=tf.float32) #[batch_size x LP x (2 hidden + HH)]
        W_P_nl = tf.get_variable("W_P_nl", [N, P_dim + HH], dtype=tf.float32)
        b_P_nl = tf.get_variable("b_P_nl", [N], dtype=tf.float32)
        timesteps_H = tf.reshape(tf.transpose(tf.to_float(timesteps_H), perm=[2,0,1]), [P_dim + HH, -1]) #[2 * hidden_size + HH, batch_size * NUM_FEATS]
        y_P_att = tf.tanh(tf.transpose(tf.matmul(W_P_nl, timesteps_H)) + b_P_nl) # [batch_size x LP, N]
        W_P_nl2 = tf.get_variable("W_P_nl2", [N, P_dim + HH], dtype=tf.float32)
        b_P_nl2 = tf.get_variable("b_P_nl2", [N], dtype=tf.float32)
        g_P_att = tf.nn.sigmoid(tf.transpose(tf.matmul(W_P_nl2, timesteps_H)) + b_P_nl2) # [batch_size x LP, N]
        y_P = tf.multiply(y_P_att, g_P_att)  # [batch_size x LP, N]
        W_P_a = tf.get_variable("W_P_a", [1, FEAT_SIZE], dtype=tf.float32)
        a_P = tf.matmul(W_P_a, tf.transpose(y_P)) # [1, batch_size x LP]
        #...to obtain a scalar attention weight α_{i,t}
        alpha_P = tf.nn.softmax(tf.transpose(a_P)) # [1, batch_size x LP]
        alpha_P = tf.reshape(alpha_P, [options.batch_size, options.max_len_p, 1])
        #The image features are then weighted by the normalized values and summed
        timesteps_att = tf.map_fn(lambda x: tf.multiply(x[1], x[0]), (output, alpha_P), dtype=alpha_P.dtype) #[batch_size x LP x 2_hidden]
        timesteps_att = tf.map_fn(lambda x: tf.reduce_sum(x, 0), timesteps_att) #[batch_size, 2 x hidden]


    #The representations of the question (q) and of the image (v̂) are passed through non-linear layers...
    W_h_nl = tf.get_variable("W_h_nl", [options.hidden_size, HH], dtype=tf.float32)
    b_h_nl = tf.get_variable("b_h_nl", [options.hidden_size], dtype=tf.float32)
    y_h_att = tf.tanh(tf.transpose(tf.matmul(W_h_nl, tf.transpose(H_embedding))) + b_h_nl) # [batch_size x H]

    W_h_nl2 = tf.get_variable("W_h_nl2", [options.hidden_size, HH], dtype=tf.float32)
    b_h_nl2 = tf.get_variable("b_h_nl2", [options.hidden_size], dtype=tf.float32)
    g_h_att = tf.nn.sigmoid(tf.transpose(tf.matmul(W_h_nl2, tf.transpose(H_embedding))) + b_h_nl2) # [batch_size x H]
    y_h = tf.multiply(y_h_att, g_h_att)  # [batch_size x H]


    W_I_nl = tf.get_variable("W_I_nl", [options.hidden_size, FEAT_SIZE], dtype=tf.float32)
    b_I_nl = tf.get_variable("b_I_nl", [options.hidden_size], dtype=tf.float32)
    y_I_att = tf.tanh(tf.transpose(tf.matmul(W_I_nl, tf.transpose(features_att))) + b_I_nl) # [batch_size x FEAT_SIZE]

    W_I_nl2 = tf.get_variable("W_I_nl2", [options.hidden_size, FEAT_SIZE], dtype=tf.float32)
    b_I_nl2 = tf.get_variable("b_I_nl2", [options.hidden_size], dtype=tf.float32)
    g_I_att = tf.nn.sigmoid(tf.transpose(tf.matmul(W_I_nl2, tf.transpose(features_att))) + b_I_nl2) # [batch_size x FEAT_SIZE]
    y_I = tf.multiply(y_I_att, g_I_att)  # [batch_size x H]


    if options.with_P_top_down:
        W_P_nl3 = tf.get_variable("W_P_nl3", [options.hidden_size, P_dim], dtype=tf.float32)
        b_P_nl3 = tf.get_variable("b_P_nl3", [options.hidden_size], dtype=tf.float32)
        y_P_att3 = tf.tanh(tf.transpose(tf.matmul(W_P_nl3, tf.transpose(timesteps_att))) + b_P_nl3) # [batch_size x H]

        W_P_nl4 = tf.get_variable("W_P_nl4", [options.hidden_size, P_dim], dtype=tf.float32)
        b_P_nl4 = tf.get_variable("b_P_nl4", [options.hidden_size], dtype=tf.float32)
        g_P_att4 = tf.nn.sigmoid(tf.transpose(tf.matmul(W_P_nl4, tf.transpose(timesteps_att))) + b_P_nl4) # [batch_size x H]
        y_P4 = tf.multiply(y_P_att3, g_P_att4)  # [batch_size x H]


    #...and then combined with a simple Hadamard product
    fusion = tf.multiply(y_h, y_I) # [batch_size x H]

    if options.with_mlp:
        mlp1 = mlp(fusion, options.hidden_size, NUM_CLASSES, name="mlp1")
        mlp2 = mlp(fusion, options.hidden_size, NUM_CLASSES, name="mlp2")
        score = mlp1 + mlp2
    else:
        if options.with_P_top_down:
            fusion = tf.multiply(fusion, y_P4) # [batch_size x H]

        #Passes the joint embedding through a non-linear layer f_o...
        W_fusion_nl = tf.get_variable("W_fusion_nl", [FEAT_SIZE / 2, options.hidden_size], dtype=tf.float32)
        b_fusion_nl = tf.get_variable("b_fusion_nl", [FEAT_SIZE / 2], dtype=tf.float32)
        y_fusion_att = tf.tanh(tf.transpose(tf.matmul(W_fusion_nl, tf.transpose(fusion))) + b_fusion_nl) # [batch_size x FEAT_SIZE / 2]

        W_fusion_nl2 = tf.get_variable("W_fusion_nl2", [FEAT_SIZE / 2, options.hidden_size], dtype=tf.float32)
        b_fusion_nl2 = tf.get_variable("b_fusion_nl2", [FEAT_SIZE / 2], dtype=tf.float32)
        g_fusion_att = tf.nn.sigmoid(tf.transpose(tf.matmul(W_fusion_nl2, tf.transpose(fusion))) + b_fusion_nl2) # [batch_size x FEAT_SIZE / 2]
        y_fusion = tf.multiply(y_fusion_att, g_fusion_att)  # [batch_size x FEAT_SIZE / 2]

        #y_fusion = tf.concat([P_embedding, y_fusion], 1) # [batch_size x (FEAT_SIZE / 2 + HH)]

        #...then through a linear mapping w_o to predict a score ŝ for each of the 3 candidates
        #W_o = tf.get_variable("W_o", [NUM_CLASSES, FEAT_SIZE / 2], dtype=tf.float32)
        #score = tf.transpose(tf.matmul(W_o, tf.transpose(y_fusion))) # [batch_size x NUM_CLASSES]

        dropout_prob = 0.5
        dimension = int(FEAT_SIZE / 2) #int(FEAT_SIZE / 2 + options.hidden_size * 2)

        gated_first_layer = tf.nn.dropout(gated_tanh(y_fusion, dimension), keep_prob=dropout_prob)
        gated_second_layer = tf.nn.dropout(gated_tanh(gated_first_layer, dimension), keep_prob=dropout_prob)
        gated_third_layer = tf.nn.dropout(gated_tanh(gated_second_layer, dimension), keep_prob=dropout_prob)

        score = tf.contrib.layers.fully_connected(gated_third_layer, NUM_CLASSES, activation_fn=None)

    prob = tf.nn.softmax(score)
    gold_matrix = tf.one_hot(labels, NUM_CLASSES, dtype=tf.float32)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prob, labels=gold_matrix))
    clipper = 50


    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    tvars = tf.trainable_variables()
    l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tvars if v.get_shape().ndims > 1])
    loss = loss + 1e-5 * l2_loss
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), clipper)
    if options.decay:
        optimizer = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)
    else:
        optimizer = optimizer.apply_gradients(zip(grads, tvars))

    loss_summ = tf.summary.scalar('loss', loss)
    train_summary.append(loss_summ)
