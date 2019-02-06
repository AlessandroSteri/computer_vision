import os
import math
import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm
from gte.preprocessing.batch import generate_batch
from gte.info import TB_DIR, NUM_CLASSES, DEV_DATA, TRAIN_DATA, TEST_DATA, TEST_DATA_HARD, BEST_F1, LEN_TRAIN, LEN_DEV, LEN_TEST, LEN_TEST_HARD, NUM_FEATS, FEAT_SIZE
from gte.utils.tf import bilstm_layer, highway
from gte.match.match_utils import bilateral_match_func
from gte.images.image2vec import Image2vec
from gte.att.attention import multihead_attention

class GroundedTextualEntailmentModel(object):
    """Model for Grounded Textual Entailment."""
    def __init__(self, options, ID, embeddings, word2id, id2word, label2id, id2label):
        self.options = options
        self.ID = ID
        self.embeddings = embeddings
        self.word2id = word2id
        self.id2word = id2word
        self.label2id = label2id
        self.id2label = id2label
        self.train_summary = []
        self.eval_summary = []
        self.test_summary = []
        self.graph = self.create_graph()
        self.session = tf.Session()
        self.best_f1 = self.get_best_f1()
        self.img2vec = Image2vec() if options.with_img else None

    def __enter__(self):
        # self.session = tf.Session()
        return self

    def __exit__(self, exc_type, exc_val, traceback):
        if self.session is not None:
            self.session.close()

    def get_session(self):
        if self.session is None:
            self.session = tf.Session()
        return self.session

    def restore_session(self, model_ckpt):
        self.saver.restore(self.session, model_ckpt)
        print("[GTE][MODEL] Model restored from {}.".format(model_ckpt))

    def get_best_f1(self):
        if not os.path.exists(BEST_F1): return 0.50
        with open(BEST_F1, 'r') as f:
            return float(f.readline())

    def store_best_f1(self, f1):
        with open(BEST_F1, 'w') as f:
            f.write('{}'.format(f1))

    def create_graph(self):
        print('[GTE][MODEL]Creating graph...')
        if not self.options.use_gpu:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

        # graph = tf.Graph()
        # with graph.as_default():

        if self.options.dev_env:
            tf.set_random_seed(1)

        self.input_layer()
        self.embedding_layer()
        self.context_layer()
        if self.options.with_matching: self.matching_layer()
        # self.aggregation_layer()
        self.opt_loss_layer()
        self.prediction_layer()

        self.create_evaluation_graph()

        self.saver = tf.train.Saver()
        print('[GTE][MODEL] Created graph!')
        # return graph

    def input_layer(self):
        # Define input data tensors.
        with tf.name_scope('INPUTS'):
            self.P = tf.placeholder(tf.int32, shape=[self.options.batch_size, self.options.max_len_p], name='P')  # shape [batch_size, max_len_p]
            self.H = tf.placeholder(tf.int32, shape=[self.options.batch_size, self.options.max_len_h], name='H')  # shape [batch_size, max_len_h]
            self.I = tf.placeholder(tf.float32, shape=[self.options.batch_size, NUM_FEATS, FEAT_SIZE], name='H')  # shape [batch_size, max_len_h]
            self.labels = tf.placeholder(tf.int32, shape=[self.options.batch_size], name='Labels')  # shape [batch_size]
            self.lengths_P = tf.placeholder(tf.int32, shape=[self.options.batch_size], name='Lengths_P')  # shape [batch_size]
            self.lengths_H = tf.placeholder(tf.int32, shape=[self.options.batch_size], name='Lengths_H')  # shape [batch_size]
        with tf.name_scope('INPUTS_MASKS'):
            self.P_mask = tf.sequence_mask(self.lengths_P, self.options.max_len_p, dtype=tf.float32, name='P_mask')  # [batch_size, max_len_p]
            self.H_mask = tf.sequence_mask(self.lengths_H, self.options.max_len_h, dtype=tf.float32, name='H_mask')  # [batch_size, max_len_h]
            self.I_mask = tf.sequence_mask([NUM_FEATS for _ in range(self.options.batch_size)], NUM_FEATS, dtype=tf.float32, name='I_mask')  # [batch_size, NUM_FEATS]
        with tf.name_scope('IS_TRAINING'):
            self.is_training = tf.placeholder(tf.bool, shape=[], name='is_training')
            if self.options.dropout:
                self.keep_probability = tf.placeholder(tf.float32, shape=[], name='un_dropout_rate')

    def embedding_layer(self):
        with tf.name_scope('EMBEDDINGS'):
            embeddings = tf.Variable(self.embeddings, dtype=tf.float32, trainable=self.options.trainable, name='embeddings')

            # selectors
            self.P_lookup = tf.nn.embedding_lookup(embeddings, self.P, name='P_lookup')  # shape (batch_size, max_len_p, embedding_size)
            self.H_lookup = tf.nn.embedding_lookup(embeddings, self.H, name='H_lookup')  # shape (batch_size, max_len_h, embedding_size)

    def context_layer(self):
        with tf.name_scope('CONTEXT'):
            # import ipdb; ipdb.set_trace()  # TODO BREAKPOINT
            B = self.options.batch_size
            HH = 2*self.options.hidden_size
            L_p = self.options.max_len_p
            L_h = self.options.max_len_h
            # self.context_p = bilstm_layer(self.P_lookup, self.lengths_P, self.options.hidden_size, name='BILSTM_P')
            kp = self.keep_probability if self.options.dropout else 1
            self.context_p = self.directional_lstm(self.P_lookup, 2, self.options.hidden_size, kp)
            # self.context_h = bilstm_layer(self.H_lookup, self.lengths_H, self.options.hidden_size, name='BILSTM_H')
            self.context_h = self.directional_lstm(self.H_lookup, 2, self.options.hidden_size, kp)

    def matching_layer(self):
        with tf.name_scope('MATCHING'):
            # ========Bilateral Matching=====
            out_image_feats = None
            in_question_repres = self.context_p
            in_passage_repres = self.context_h
            in_question_dep_cons = None
            in_passage_dep_cons = None
            self.question_lengths = self.lengths_P
            self.passage_lengths = self.lengths_H
            question_mask = self.P_mask
            mask = self.H_mask
            MP_dim = 10
            input_dim = 2*self.options.hidden_size
            with_filter_layer = False
            context_layer_num = 1
            context_lstm_dim = self.options.hidden_size
            is_training = True
            dropout_rate = 0.5
            with_match_highway = False
            aggregation_layer_num = 1
            aggregation_lstm_dim = 300  # self.options.hidden_size
            highway_layer_num = 1
            with_aggregation_highway = True
            with_lex_decomposition = False
            lex_decompsition_dim = -1
            with_full_match = True
            with_maxpool_match = True
            with_attentive_match = True
            with_max_attentive_match = True
            with_left_match = True
            with_right_match = True
            with_dep = False
            with_image = False
            with_mean_aggregation = True
            image_with_hypothesis_only = True
            with_img_attentive_match = True
            with_img_full_match = True
            with_img_maxpool_match = False
            with_img_max_attentive_match = True
            image_context_layer = False
            img_dim = 100

            (self.match_representation, self.match_dim) = bilateral_match_func(out_image_feats,
                                                                     in_question_repres,
                                                                     in_passage_repres,
                                                                     in_question_dep_cons,
                                                                     in_passage_dep_cons,
                                                                     self.question_lengths,
                                                                     self.passage_lengths,
                                                                     question_mask,
                                                                     mask,
                                                                     MP_dim,
                                                                     input_dim,
                                                                     with_filter_layer,
                                                                     context_layer_num,
                                                                     context_lstm_dim,
                                                                     is_training,dropout_rate,
                                                                     with_match_highway,
                                                                     aggregation_layer_num,
                                                                     aggregation_lstm_dim,
                                                                     highway_layer_num,
                                                                     with_aggregation_highway,
                                                                     with_lex_decomposition,
                                                                     lex_decompsition_dim,
                                                                     with_full_match,
                                                                     with_maxpool_match,
                                                                     with_attentive_match,
                                                                     with_max_attentive_match,
                                                                     with_left_match,
                                                                     with_right_match,
                                                                     with_dep=with_dep,
                                                                     with_image=with_image,
                                                                     with_mean_aggregation=with_mean_aggregation,
                                                                     image_with_hypothesis_only=image_with_hypothesis_only,
                                                                     with_img_attentive_match=with_img_attentive_match,
                                                                     with_img_full_match=with_img_full_match,
                                                                     with_img_maxpool_match=with_img_maxpool_match,
                                                                     with_img_max_attentive_match=with_img_max_attentive_match,
                                                                     image_context_layer=image_context_layer,
                                                                     img_dim=img_dim)

            #========Prediction Layer=========
            w_0 = tf.get_variable("w_0", [self.match_dim, self.match_dim/2], dtype=tf.float32)
            b_0 = tf.get_variable("b_0", [self.match_dim/2], dtype=tf.float32)
            w_1 = tf.get_variable("w_1", [self.match_dim/2, 3],dtype=tf.float32)
            b_1 = tf.get_variable("b_1", [3],dtype=tf.float32)

            logits = tf.matmul(self.match_representation, w_0) + b_0
            self.match_logits = tf.tanh(logits)
            self.match_logits = tf.matmul(self.match_logits, w_1) + b_1

    def opt_loss_layer(self):
        with tf.name_scope('OPT_LOSS'):

            if not self.options.with_matching:
                B = self.options.batch_size
                HH = 2*self.options.hidden_size
                L_p = self.options.max_len_p
                L_h = self.options.max_len_h
                self.latent_repr = tf.concat([self.context_p, self.context_h], axis=-2, name='latent_repr')
                if self.options.with_img:
                    self.latent_repr = highway(self.latent_repr, HH, tf.nn.relu)
                    self.latent_repr_flat = tf.reshape(self.latent_repr, [B * (L_p + L_h) , HH], name='lrf')
                    W_img = tf.get_variable("w_img", [HH, FEAT_SIZE], dtype=tf.float32)
                    b_img = tf.get_variable("b_img", [FEAT_SIZE], dtype=tf.float32)
                    self.latent_repr_flat = tf.matmul(self.latent_repr_flat, W_img) + b_img
                    self.latent_repr = tf.reshape(self.latent_repr_flat, [B, (L_p + L_h), FEAT_SIZE], name='ciaociao')
                    self.latent_repr = tf.concat([self.latent_repr, self.I], axis=-2, name='latent_repr') # [B, LP+LH+NUM_FEATS, FEAT_SIZE]
                    self.latent_repr = highway(self.latent_repr, FEAT_SIZE, tf.nn.relu)
                    self.latent_repr_flat = tf.reshape(self.latent_repr, [B, (L_p + L_h + NUM_FEATS) * FEAT_SIZE], name='ciaociaociaociao')
                    W = tf.get_variable("w", [(L_p + L_h + NUM_FEATS) * FEAT_SIZE, NUM_CLASSES], dtype=tf.float32)
                if self.options.with_img2:
                    self.context_p = highway(self.context_p, HH, tf.nn.relu)
                    self.context_h = highway(self.context_h, HH, tf.nn.relu)
                    latent_repr_flat_p = tf.reshape(self.context_p, [B * L_p, HH], name='lrf_p')
                    latent_repr_flat_h = tf.reshape(self.context_h, [B * L_h, HH], name='lrf_h')
                    # linear map from HH to FEAT_SIZE uniwue for p and h
                    W_ph = tf.get_variable("w_pi", [HH, FEAT_SIZE], dtype=tf.float32)
                    b_ph = tf.get_variable("b_pi", [FEAT_SIZE], dtype=tf.float32)
                    latent_repr_flat_p = tf.matmul(latent_repr_flat_p, W_ph) + b_ph
                    latent_repr_flat_h = tf.matmul(latent_repr_flat_h, W_ph) + b_ph
                    latent_repr_flat_p = tf.reshape(latent_repr_flat_p, [B, L_p, FEAT_SIZE], name='lrf_p_3d')
                    latent_repr_flat_h = tf.reshape(latent_repr_flat_h, [B, L_h, FEAT_SIZE], name='lrf_h_3d')
                    latent_repr_flat_pi = tf.concat([self.I, latent_repr_flat_p], axis=-2, name='lrf_IP_3d') # [B, LP+NUM_FEATS, FEAT_SIZE]
                    latent_repr_flat_hi = tf.concat([self.I, latent_repr_flat_h], axis=-2, name='lrf_IH_3d') # [B, LH+NUM_FEATS, FEAT_SIZE]
                    self.latent_repr_PI = highway(latent_repr_flat_pi, FEAT_SIZE, tf.nn.relu) # [B, LP+NUM_FEATS, FEAT_SIZE]
                    self.latent_repr_HI = highway(latent_repr_flat_hi, FEAT_SIZE, tf.nn.relu) # [B, LH+NUM_FEATS, FEAT_SIZE]
                    if self.options.attentive:
                        self.latent_repr_PI = multihead_attention(self.latent_repr_PI, self.latent_repr_PI, is_training=self.is_training, scope="PI_attention")
                        self.latent_repr_HI = multihead_attention(self.latent_repr_HI, self.latent_repr_HI, is_training=self.is_training, scope="HI_attention")
                    if self.options.attentive_swap:
                        self.latent_repr_PI = multihead_attention(self.latent_repr_PI, self.latent_repr_HI, is_training=self.is_training, scope="PI_attention")
                        self.latent_repr_HI = multihead_attention(self.latent_repr_HI, self.latent_repr_PI, is_training=self.is_training, scope="HI_attention")

                    self.latent_repr = tf.concat([latent_repr_flat_pi, latent_repr_flat_hi], axis=-2, name='latent_repr') # [B, LP+LH+NUM_FEATS, FEAT_SIZE]
                    self.latent_repr = highway(self.latent_repr, FEAT_SIZE, tf.nn.relu)
                    self.latent_repr_flat = tf.reshape(self.latent_repr, [B, (L_p + L_h + 2*NUM_FEATS) * FEAT_SIZE], name='ciaociaociaociao')
                    if self.options.dropout:
                        self.latent_repr_flat = tf.nn.dropout(self.latent_repr_flat, self.keep_probability)
                    W = tf.get_variable("w", [(L_p + L_h + 2*NUM_FEATS) * FEAT_SIZE, NUM_CLASSES], dtype=tf.float32)

                else:
                    self.latent_repr_flat = tf.reshape(self.latent_repr, [B, (L_p + L_h) * HH], name='lrf')
                    W = tf.get_variable("w", [(L_p + L_h) * HH, NUM_CLASSES], dtype=tf.float32)

                b = tf.get_variable("b", [NUM_CLASSES], dtype=tf.float32)



                # W_match = tf.get_variable("w_match", [self.match_dim/2, NUM_CLASSES], dtype=tf.float32)
                # b_match = tf.get_variable("b_match", [NUM_CLASSES], dtype=tf.float32)
                # self.pred = tf.matmul(self.match_logits, W_match) + b_match

                self.pred = tf.matmul(self.latent_repr_flat, W) + b

                self.score = self.pred

                # gold_matrix = tf.one_hot(self.labels, NUM_CLASSES, dtype=tf.float32)

                # self.losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.score, labels=gold_matrix, name='unmasked_losses')
                # self.loss = tf.reduce_mean(self.losses)
                # self.optimizer = tf.train.AdamOptimizer(self.options.learning_rate).minimize(self.loss) # TODO DA QUI VIENE INDEXSLICES


                self.prob = tf.nn.softmax(self.score)
                gold_matrix = tf.one_hot(self.labels, NUM_CLASSES, dtype=tf.float32)
                self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.score, labels=gold_matrix))
                clipper = 50
                optimizer = tf.train.AdamOptimizer(learning_rate=self.options.learning_rate)
                tvars = tf.trainable_variables()
                l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tvars if v.get_shape().ndims > 1])
                self.loss = self.loss + 1e-5 * l2_loss
                grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), clipper)
                self.optimizer = optimizer.apply_gradients(zip(grads, tvars))
            else:

                self.prob = tf.nn.softmax(self.match_logits)
                gold_matrix = tf.one_hot(self.labels, NUM_CLASSES, dtype=tf.float32)
                self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.match_logits, labels=gold_matrix))

                clipper = 50
                optimizer = tf.train.AdamOptimizer(learning_rate=self.options.learning_rate)
                tvars = tf.trainable_variables()
                l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tvars if v.get_shape().ndims > 1])
                self.loss = self.loss + 1e-5 * l2_loss
                grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), clipper)
                self.optimizer = optimizer.apply_gradients(zip(grads, tvars))

            loss_summ = tf.summary.scalar('loss', self.loss)
            self.train_summary.append(loss_summ)

    def prediction_layer(self):
        with tf.name_scope('PREDICTION'):

            if self.options.with_matching: self.score = self.match_logits
            self.softmax_score = tf.nn.softmax(self.score, name='softmax_score')
            self.predict_op = tf.cast(tf.argmax(self.softmax_score, axis=-1), tf.int32, name='predict_op')

    def aggregation_layer(self):
        with tf.name_scope('AGGREGATION'):
            pass

    def create_evaluation_graph(self):
        with tf.name_scope('eval'):
            # self.precision = tf.placeholder(tf.float32, [])
            # self.recall = tf.placeholder(tf.float32, [])
            self.accuracy = tf.placeholder(tf.float32, [])
            self.f1 = tf.placeholder(tf.float32, [])
            # self.eval_summary.append(tf.summary.scalar('Precision', self.precision))
            # self.eval_summary.append(tf.summary.scalar('Recall', self.recall))
            self.eval_summary.append(tf.summary.scalar('Accuracy', self.accuracy))
            self.eval_summary.append(tf.summary.scalar('F1', self.f1))

    def predict(self, DATASET):
        predictions, labels = [], []
        for batch in tqdm(generate_batch(DATASET,
                                         self.options.batch_size,
                                         self.word2id,
                                         self.label2id,
                                         img2vec=self.img2vec,
                                         max_len_p=self.options.max_len_p,
                                         max_len_h=self.options.max_len_h),
                          total=math.ceil(LEN_DEV / self.options.batch_size)):
            if batch is None: break
            feed_dict = {self.P: batch.P,
                         self.H: batch.H,
                         self.I: batch.I,
                         self.lengths_P: batch.lengths_P,
                         self.lengths_H: batch.lengths_H}
            feed_dict[self.is_training] = False
            if self.options.dropout:
                feed_dict[self.keep_probability] = 1
            [p] = self.session.run([self.predict_op], feed_dict=feed_dict)
            predictions += [_ for _ in p]
            labels += [_ for _ in batch.labels]

        return np.array(predictions), np.array(labels)

    def fit(self, evaluate=False, test=False):
        print("RunID: {}".format(self.ID))
        tensorboard_dir = os.path.join(TB_DIR, self.ID)
        # with tf.Session() as session:
        session = self.session
        self.writer = tf.summary.FileWriter(tensorboard_dir, session.graph)
        # init variables
        session.run(tf.global_variables_initializer())
        session.run(tf.local_variables_initializer())
        step = -1
        for _epoch in range(self.options.epoch):
            epoch = _epoch + 1
            print('Starting epoch: {}/{}'.format(epoch, self.options.epoch))
            # import ipdb; ipdb.set_trace()  # TODO BREAKPOINT
            for iteration, batch in tqdm(enumerate(generate_batch(TRAIN_DATA,
                                                                  self.options.batch_size,
                                                                  self.word2id,
                                                                  self.label2id,
                                                                  img2vec=self.img2vec,
                                                                  max_len_p=self.options.max_len_p,
                                                                  max_len_h=self.options.max_len_h)),
                                         total=math.ceil(LEN_TRAIN / self.options.batch_size)):
                # import ipdb; ipdb.set_trace()  # TODO BREAKPOINT
                if batch is None:
                    print("End of eval.")
                    break
                step += 1
                run_metadata = tf.RunMetadata()
                train_summary = tf.summary.merge(self.train_summary) # TODO should be outside loop?
                feed_dict = {self.P: batch.P,
                             self.H: batch.H,
                             self.I: batch.I,
                             self.labels: batch.labels,
                             self.lengths_P: batch.lengths_P,
                             self.lengths_H: batch.lengths_H}
                feed_dict[self.is_training] = True
                if self.options.dropout:
                    feed_dict[self.keep_probability] = 0.5

                result = session.run([self.optimizer,
                                      self.loss,
                                      self.predict_op,
                                      # self.match_representation,
                                      # self.match_logits,
                                      # self.match_dim,
                                      train_summary],
                                     feed_dict=feed_dict,
                                     run_metadata=run_metadata)
                _, loss, predictions, summary= result
                # import ipdb; ipdb.set_trace()  # TODO BREAKPOINT

                # Add returned summaries to writer in each step, where
                self.writer.add_summary(summary, step)

                if step % self.options.step_check == 0:
                    print("RunID:", self.ID)
                    print("Epoch:", epoch, "Iteration:", iteration, "Global Iteration:", step)
                    print("Loss: ", loss)
                    if step != 0:
                        if step >= 1000: # use module when is not crushing
                            # Predict without training over test sets
                            test_predictions, test_labels = self.predict(TEST_DATA)
                            test_HARD_predictions, test_HARD_labels = self.predict(TEST_DATA_HARD)
                            assert len(test_labels) == len(test_predictions)
                            assert len(test_HARD_labels) == len(test_HARD_predictions)
                            # import ipdb; ipdb.set_trace()  # TODO BREAKPOINT
                            accuracy_test = sum(test_labels == test_predictions)/len(test_labels)
                            accuracy_test_HARD = sum(test_HARD_labels == test_HARD_predictions)/len(test_HARD_labels)
                            print("TEST Accuracy: {}".format(accuracy_test))
                            print("TEST_HARD Accuracy: {}".format(accuracy_test_HARD))


                        # Predict without training
                        eval_predictions, labels = self.predict(DEV_DATA)
                        assert len(labels) == len(eval_predictions)
                        accuracy = sum(labels == eval_predictions)/len(labels)
                        print("Accuracy: {}".format(accuracy))
                        f1 = f1_score(labels, eval_predictions, average='micro')
                        print("F1: {}".format(f1))
                        print('--------------------------------')

                        delta = 1.1
                        if f1 >= self.best_f1 * delta:
                            print("New Best F1: {}, old was: {}".format(f1, self.best_f1))
                            print('--------------------------------')
                            self.store_best_f1(f1)
                            path = self.saver.save(session, os.path.join(tensorboard_dir, '{}_model.ckpt'.format(f1)))

                        feed_dictionary = {self.accuracy: accuracy,
                                           self.f1: f1}

                        eval_summary = tf.summary.merge(self.eval_summary)
                        eval_summ = session.run(eval_summary,
                                                feed_dict=feed_dictionary)
                        self.writer.add_summary(eval_summ, step)
        self.writer.close()

    def directional_lstm(self, input_data, num_layers, rnn_size, keep_prob):
        output = input_data
        for layer in range(num_layers):
            with tf.variable_scope('encoder_{}'.format(layer),reuse=tf.AUTO_REUSE):
                cell_fw = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.truncated_normal_initializer(-0.1, 0.1))
                # cell_fw = tf.contrib.rnn.AttentionCellWrapper(cell_fw, attn_length=40, state_is_tuple=True)
                # cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, input_keep_prob = keep_prob)
                cell_bw = tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.truncated_normal_initializer(-0.1, 0.1))
                # cell_bw = tf.contrib.rnn.AttentionCellWrapper(cell_bw, attn_length=40, state_is_tuple=True)
                # cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, input_keep_prob = keep_prob)
                outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, output, dtype=tf.float32, swap_memory=True)
                output = tf.concat(outputs,2)
                return output

