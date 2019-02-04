import os
import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm
from gte.preprocessing.batch import generate_batch
from gte.info import TB_DIR, NUM_CLASSES, DEV_DATA, TRAIN_DATA, BEST_F1
from gte.utils.tf import bilstm_layer
from gte.match.match_utils import bilateral_match_func

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
        self.matching_layer()
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
            self.labels = tf.placeholder(tf.int32, shape=[self.options.batch_size], name='Labels')  # shape [batch_size]
            self.lengths_P = tf.placeholder(tf.int32, shape=[self.options.batch_size], name='Lengths_P')  # shape [batch_size]
            self.lengths_H = tf.placeholder(tf.int32, shape=[self.options.batch_size], name='Lengths_H')  # shape [batch_size]

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
            self.context_p = bilstm_layer(self.P_lookup, self.lengths_P, self.options.hidden_size, name='BILSTM_P')
            self.context_h = bilstm_layer(self.H_lookup, self.lengths_H, self.options.hidden_size, name='BILSTM_H')

    def matching_layer(self):
        with tf.name_scope('MATCHING'):
            P_mask = tf.sequence_mask(self.lengths_P, self.options.max_len_p, dtype=tf.float32) # [batch_size, max_len_p]
            H_mask = tf.sequence_mask(self.lengths_H, self.options.max_len_h, dtype=tf.float32) # [batch_size, max_len_h]
            # ========Bilateral Matching=====
            out_image_feats = None
            in_question_repres = self.context_p
            in_passage_repres = self.context_h
            in_question_dep_cons = None
            in_passage_dep_cons = None
            self.question_lengths = self.lengths_P
            self.passage_lengths = self.lengths_H
            question_mask = P_mask
            mask = H_mask
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

    def opt_loss_layer(self):
        with tf.name_scope('OPT_LOSS'):
            W_match = tf.get_variable("w_match", [self.match_dim/2, NUM_CLASSES], dtype=tf.float32)
            b_match = tf.get_variable("b_match", [NUM_CLASSES], dtype=tf.float32)

            # self.latent_repr = tf.concat([self.context_p, self.context_h], axis=-2, name='latent_repr')
            # self.latent_repr_flat = tf.reshape(self.latent_repr, [B, (L_p + L_h) * HH], name='lrf')

            # W = tf.get_variable("w", [(L_p + L_h) * HH, NUM_CLASSES], dtype=tf.float32)
            # b = tf.get_variable("b", [NUM_CLASSES], dtype=tf.float32)

            self.pred = tf.matmul(self.match_logits, W_match) + b_match
            self.score = self.pred
            self.losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.score, labels=self.labels, name='unmasked_losses')
            self.loss = tf.reduce_mean(self.losses)
            self.optimizer = tf.train.AdamOptimizer(self.options.learning_rate).minimize(self.loss) # TODO DA QUI VIENE INDEXSLICES
            loss_summ = tf.summary.scalar('loss', self.loss)
            self.train_summary.append(loss_summ)

    def prediction_layer(self):
        with tf.name_scope('PREDICTION'):
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
                                         max_len_p=self.options.max_len_p,
                                         max_len_h=self.options.max_len_h)):
            if batch is None: break
            feed_dict_train = {self.P: batch.P,
                               self.H: batch.H,
                               self.lengths_P: batch.lengths_P,
                               self.lengths_H: batch.lengths_H}
            [p] = self.session.run([self.predict_op], feed_dict=feed_dict_train)
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
            for iteration, batch in tqdm(enumerate(generate_batch(TRAIN_DATA, self.options.batch_size, self.word2id, self.label2id, max_len_p=self.options.max_len_p, max_len_h=self.options.max_len_h))):
                # import ipdb; ipdb.set_trace()  # TODO BREAKPOINT
                if batch is None:
                    print("End of eval.")
                    break
                step += 1
                run_metadata = tf.RunMetadata()
                train_summary = tf.summary.merge(self.train_summary) # TODO should be outside loop?
                feed_dict_train = {self.P: batch.P,
                                   self.H: batch.H,
                                   self.labels: batch.labels,
                                   self.lengths_P: batch.lengths_P,
                                   self.lengths_H: batch.lengths_H}

                result = session.run([self.optimizer,
                                      self.loss,
                                      self.predict_op,
                                      # self.match_representation,
                                      self.match_logits,
                                      # self.match_dim,
                                      train_summary],
                                     feed_dict=feed_dict_train,
                                     run_metadata=run_metadata)
                _, loss, predictions, mr, summary= result
                # import ipdb; ipdb.set_trace()  # TODO BREAKPOINT

                # Add returned summaries to writer in each step, where
                self.writer.add_summary(summary, step)

                if step % self.options.step_check == 0:
                    print("RunID:", self.ID)
                    print("Epoch:", epoch, "Iteration:", iteration, "Global Iteration:", step)
                    print("Loss: ", loss)
                    print('--------------------------------')
                    if step != 0:
                        # give back control to callee to run evaluation
                        eval_predictions, labels = self.predict(DEV_DATA)
                        assert len(labels) == len(eval_predictions)
                        accuracy = sum(labels == eval_predictions)/len(labels)
                        print("Accuracy: {}".format(accuracy))
                        f1 = f1_score(labels, eval_predictions, average='micro')
                        print("F1: {}".format(f1))

                        delta = 1.1
                        if f1 >= self.best_f1 * delta:
                            path = self.saver.save(session, os.path.join(tensorboard_dir, '{}_model.ckpt'.format(f1)))
                            self.store_best_f1(f1)

                        feed_dictionary = {self.accuracy: accuracy,
                                           self.f1: f1}

                        eval_summary = tf.summary.merge(self.eval_summary)
                        eval_summ = session.run(eval_summary,
                                                feed_dict=feed_dictionary)
                        self.writer.add_summary(eval_summ, step)
        self.writer.close()
