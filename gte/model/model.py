import os
import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm
from gte.preprocessing.batch import generate_batch
from gte.info import TB_DIR, NUM_CLASSES, DEV_DATA, TRAIN_DATA
from gte.utils.tf import bilstm_layer

class GroundedTextualEntailmentModel(object):
    """Model for Grounded Textual Entailment."""
    def __init__(self, options, ID, embeddings, word2id, id2word, label2id, id2label):
        self.options = options
        self.ID = ID
        # import ipdb; ipdb.set_trace()  # TODO BREAKPOINT
        self.embeddings = embeddings
        self.word2id = word2id
        self.id2word = id2word
        self.label2id = label2id
        self.id2label = id2label
        self.train_summary = []
        self.eval_summary = []
        self.test_summary = []
        self.graph = self.create_graph()

    def create_graph(self):
        print('Creating graph...')
        if not self.options.use_gpu:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

        ### MODEL DEFINITION ###
        graph = tf.Graph()
        with graph.as_default():
            if self.options.dev_env:
                tf.set_random_seed(1)

            self.input_layer()
            self.embedding_layer()
            self.context_layer()
            self.matching_layer()
            self.aggregation_layer()
            self.prediction_layer()
            self.opt_loss_layer()

            self.create_evaluation_graph()

            # creates the saver
            self.saver = tf.train.Saver()
        print('Created graph!')
        return graph

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
            self.latent_repr = tf.concat([self.context_p, self.context_h], axis=-2, name='latent_repr')
            self.latent_repr_flat = tf.reshape(self.latent_repr, [B, (L_p + L_h) * HH], name='lrf')

            W = tf.get_variable("w", [(L_p + L_h) * HH, NUM_CLASSES], dtype=tf.float32)
            b = tf.get_variable("b", [NUM_CLASSES], dtype=tf.float32)
            self.pred = tf.matmul(self.latent_repr_flat, W) + b
            self.score = self.pred
            self.losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.score, labels=self.labels, name='unmasked_losses')
            self.loss = tf.reduce_mean(self.losses)
            self.optimizer = tf.train.AdamOptimizer(self.options.learning_rate).minimize(self.loss) # TODO DA QUI VIENE INDEXSLICES
            loss_summ = tf.summary.scalar('loss', self.loss)
            self.train_summary.append(loss_summ)

            self.softmax_score = tf.nn.softmax(self.score, name='softmax_score')
            self.predict_op = tf.cast(tf.argmax(self.softmax_score, axis=-1), tf.int32, name='predict_op')

    def matching_layer(self):
        with tf.name_scope('MATCHING'):
            pass

    def aggregation_layer(self):
        with tf.name_scope('AGGREGATION'):
            pass

    def prediction_layer(self):
        pass

    def opt_loss_layer(self):
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

    # def eval_model(self, epoch, iteration, DATASET, session):
    #     print("Evaluate model: \nEpoch{} \nIteration{}.".format(epoch, iteration))
    #     # eval_summary = tf.summary.merge(self.eval_summary)
    #     pass

    def predict(self, DATASET, session):
        predictions, labels = [], []
        for batch in tqdm(generate_batch(DATASET,
                                         self.options.batch_size,
                                         self.word2id,
                                         self.label2id,
                                         max_len_p=self.options.max_len_p,
                                         max_len_h=self.options.max_len_h)):
            if batch is None:
                print("End of epoch.")
                break
            feed_dict_train = {self.P: batch.P,
                               self.H: batch.H,
                               self.lengths_P: batch.lengths_P,
                               self.lengths_H: batch.lengths_H}
            [p] = session.run([self.predict_op], feed_dict=feed_dict_train)
            # import ipdb; ipdb.set_trace()  # TODO BREAKPOINT
            predictions += [_ for _ in p]
            labels += [_ for _ in batch.labels]
        return np.array(predictions), np.array(labels)

    def train(self, num_epoch, evaluate=False, test=False):
        print("Training for {} epochs.".format(num_epoch))
        tensorboard_dir = os.path.join(TB_DIR, self.ID)
        with tf.Session(graph=self.graph) as session:
            self.writer = tf.summary.FileWriter(tensorboard_dir, session.graph)
            # init variables
            session.run(tf.global_variables_initializer())
            session.run(tf.local_variables_initializer())
            step = -1
            for _epoch in range(num_epoch):
                epoch = _epoch + 1
                print('Starting epoch: {}/{}'.format(epoch, num_epoch))
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
                                          train_summary],
                                         feed_dict=feed_dict_train,
                                         run_metadata=run_metadata)
                    _, loss, predictions, summary = result
                    # batch_accuracy = sum(batch.labels == predictions)/self.options.batch_size
                    # print("BATCH ACCURACY: ", batch_accuracy)
                    # print('LABELS:', batch.labels)
                    # print('PREDIC:', predictions)

                    # Add returned summaries to writer in each step, where
                    self.writer.add_summary(summary, step)

                    if step % self.options.step_check == 0:
                        print("RunID:", self.ID)
                        print("Epoch:", epoch, "Iteration:", iteration, "Global Iteration:", step)
                        print("Loss: ", loss)
                        print('--------------------------------')
                        if step != 0:
                            # give back control to callee to run evaluation
                            eval_predictions, labels = self.predict(DEV_DATA, session)
                            assert len(labels) == len(eval_predictions)
                            accuracy = sum(labels == eval_predictions)/len(labels)
                            print("Accuracy: {}".format(accuracy))
                            F1 = f1_score(labels, eval_predictions, average='micro')
                            print("F1: {}".format(F1))

                            feed_dictionary = {self.accuracy: accuracy,
                                               self.f1: F1}

                            eval_summary = tf.summary.merge(self.eval_summary)
                            eval_summ = session.run(eval_summary,
                                                    feed_dict=feed_dictionary)
                            # use num_iteration training for summary
                            self.writer.add_summary(eval_summ, step)
                            yield session, step, epoch
                            # print('ith-Epoch: {}/{}'.format(epoch, num_epoch))
            self.writer.close()

            # TODO uncomment once the model is implemented
            # Save the model for checkpoints
            # path = self.saver.save(session, os.path.join(tensorboard_dir, 'model.ckpt'))



########

    # def matching_layer(self):
    #     with tf.name_scope('MATCHING'):
    #         pass
    #         # import ipdb; ipdb.set_trace()  # TODO BREAKPOINT
    #         # context_p_flat = tf.reshape(self.context_p, [self.options.batch_size*self.options.max_len_p, 2*self.options.hidden_size], name='context_p_flat') #shape (batch*step_p, hidden)
    #         # context_h_flat = tf.reshape(self.context_h, [self.options.batch_size*self.options.max_len_h, 2*self.options.hidden_size], name='context_h_flat') #shape (batch*step_h, hidden)
    #         #TODO check if one is the other transposed
    #         # self.match_P = tf.matmul(context_p_flat, tf.transpose(context_h_flat)) #SHAPE [B*LP, B*LH]
    #         # self.match_H = tf.matmul(context_h_flat, tf.transpose(context_p_flat)) #SHAPE [B*LH, B*LP]
    #
    # def aggregation_layer(self):
    #     with tf.name_scope('AGGREGATION'):
    #         pass
    #         # aggregate_p = bilstm_layer(self.context_p, self.lengths_P, self.options.hidden_size, name='BILSTM_AGGREGATE_P')
    #         # aggregate_h = bilstm_layer(self.context_h, self.lengths_H, self.options.hidden_size, name='BILSTM_AGGREGATE_H')
    #         # self.latent_repr = tf.concat([aggregate_p, tf.transpose(aggregate_h)], axis=-1, name='latent_repr')
    #         # self.latent_repr = tf.concat([aggregate_p, aggregate_h], axis=-2, name='latent_repr')

    # def old_prediction_layer(self):
    #     with tf.name_scope('PREDICTION'):
    #         # match_dim = 2*self.options.batch_size*self.options.max_len_h #BLH
    #         match_dim = 2*self.options.hidden_size
    #         repr_flat = tf.reshape(self.latent_repr, [-1, 2*self.options.hidden_size], name='context_p_flat') #shape (batch*step_p, hidden)
    #         w_0 = tf.get_variable("w_0", [match_dim, match_dim/2], dtype=tf.float32)
    #         b_0 = tf.get_variable("b_0", [match_dim/2], dtype=tf.float32)
    #         w_1 = tf.get_variable("w_1", [match_dim/2, NUM_CLASSES],dtype=tf.float32)
    #         b_1 = tf.get_variable("b_1", [NUM_CLASSES],dtype=tf.float32)
    #
    #         logits = tf.matmul(repr_flat, w_0) + b_0
    #         logits = tf.tanh(logits)
    #         self.logits = tf.matmul(logits, w_1) + b_1
    #         self.score = tf.reshape(self.logits, [-1, self.options.max_len_p + self.options.max_len_h, NUM_CLASSES], name='score') #shape batch, step, self.out_space_size)
    #         self.softmax_score = tf.nn.softmax(self.score, name='softmax_score')
    #         self.predict = tf.cast(tf.argmax(self.softmax_score, axis=-1), tf.int32, name='predict')
    #
    #
    # def old_opt_loss_layer(self):
    #     with tf.name_scope('loss'):
    #         losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.score, labels=self.labels, name='unmasked_losses')
    #         # mask = tf.sequence_mask(self.lengths_P, name='mask')
    #         # losses = tf.boolean_mask(losses, mask, name='masked_loss')
    #         # loss_mask = tf.sequence_mask(self.sequence_lengths, self.max_sentence_len, name='mask')
    #         # loss_mask = tf.reshape(loss_mask, (self.batch_size, self.max_sentence_len))
    #         # self.loss = tf.contrib.seq2seq.sequence_loss(score, self.labels, tf.cast(loss_mask, tf.float32), name='loss')
    #         loss_mask = tf.reshape(losses, (self.options.batch_size, None, 3))
    #         self.loss = tf.contrib.seq2seq.sequence_loss(self.softmax_score, self.labels, tf.cast(loss_mask, tf.float32), name='loss')
    #
    #
    #         # Add the loss value as a scalar to summary.
    #         loss_summ = tf.summary.scalar('loss', self.loss)
    #         self.train_summary.append(loss_summ)
    #
    #
    #     with tf.name_scope('optimizer'):
    #         self.optimizer = tf.train.AdamOptimizer(self.options.learning_rate).minimize(self.loss)
