import os
import tensorflow as tf
from tqdm import tqdm
from gte.preprocessing.batch import generate_batch, iteration_per_epoch
from gte.info import TB_DIR, MAX_LEN_P, MAX_LEN_H, NUM_CLASSES
from gte.utils.tf import bilstm_layer

class GroundedTextualEntailmentModel(object):
    """Model for Grounded Textual Entailment."""
    def __init__(self, options, ID, embeddings):
        self.options = options
        self.graph = self.create_graph()
        self.ID = ID
        self.embeddings = embeddings

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
            #...
            self.embedding_layer()
            #...
            self.context_layer()
            #...
            self.prediction_layer()

            # TODO uncomment the saver once the model is implemented
            # creates the saver
            # self.saver = tf.train.Saver()
        print('Created graph!')
        return graph

    def input_layer(self):
        # Define input data tensors.
        with tf.name_scope('INPUTS'):
            self.P = tf.placeholder(tf.int32, shape=[self.options.batch_size, self.options.max_len_p], name='P')  # shape [batch_size, max_len_p]
            self.H = tf.placeholder(tf.int32, shape=[self.options.batch_size, self.options.max_len_h], name='H')  # shape [batch_size, max_len_h]
            self.labels = tf.placeholder(tf.int32, shape=[self.options.batch_size], name='Labels')  # shape [batch_size]


    def embedding_layer(self):
        with tf.name_scope('EMBEDDINGS'):
            embeddings = tf.Variable(self.embeddings, dtype=tf.float32, trainable=self.options.trainable, name='embeddings')

            # selectors
            self.P_lookup = tf.nn.embedding_lookup(embeddings, self.P, name='P_lookup')  # shape (batch_size, max_len_p, embedding_size)
            self.H_lookup = tf.nn.embedding_lookup(embeddings, self.H, name='H_lookup')  # shape (batch_size, max_len_h, embedding_size)

    def context_layer(self):
        with tf.name_scope('CONTEXT'):
            self.context_p = bilstm_layer(self.P_lookup, self.options.max_len_p, self.options.hidden_size, name='BILSTM_P')
            self.context_h = bilstm_layer(self.H_lookup, self.options.max_len_h, self.options.hidden_size, name='BILSTM_H')

    def matching_layer(self):
        with tf.name_scope('MATCHING'):
            context_p_flat = tf.reshape(self.context_p, [self.options.batch_size*self.options.max_len_p, self.options.hidden_size], name='context_p_flat') #shape (batch*step_p, hidden)
            context_h_flat = tf.reshape(self.context_h, [self.options.batch_size*self.options.max_len_h, self.options.hidden_size], name='context_h_flat') #shape (batch*step_h, hidden)
            #TODO check if one is the other transposed
            self.match_P = tf.matmul(context_p_flat, tf.transpose(context_h_flat))
            self.match_H = tf.matmul(context_h_flat, tf.transpose(context_p_flat))

    def aggregation_layer(self):
        with tf.name_scope('AGGREGATION'):
            aggregate_p = bilstm_layer(self.match_P, self.options.max_len_p, self.options.hidden_size, name='BILSTM_AGGREGATE_P')
            aggregate_h = bilstm_layer(self.match_H, self.options.max_len_h, self.options.hidden_size, name='BILSTM_AGGREGATE_H')
            self.latent_repr = tf.concat([aggregate_p, tf.transpose(aggregate_h)], axis=-1, name='latent_repr')

    def prediction_layer(self):
        with tf.name_scope('PREDICTION'):
            match_dim = 2*self.options.batch_size*self.options.max_len_h #BLH
            w_0 = tf.get_variable("w_0", [match_dim, match_dim/2], dtype=tf.float32)
            b_0 = tf.get_variable("b_0", [match_dim/2], dtype=tf.float32)
            w_1 = tf.get_variable("w_1", [match_dim/2, NUM_CLASSES],dtype=tf.float32)
            b_1 = tf.get_variable("b_1", [NUM_CLASSES],dtype=tf.float32)

            logits = tf.matmul(self.latent_repr, w_0) + b_0
            logits = tf.tanh(logits)
            # if is_training:
            #     logits = tf.nn.dropout(logits, (1 - dropout_rate))
            # else:
            #     logits = tf.multiply(logits, (1 - dropout_rate))
            logits = tf.matmul(logits, w_1) + b_1
            softmax_score = tf.nn.softmax(logits, name='softmax_score')
            self.predict = tf.cast(tf.argmax(softmax_score, axis=-1), tf.int32, name='predict')

    def opt_loss_layer(self):
        with tf.name_scope('loss'):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.labels, name='unmasked_losses')
            # mask = tf.sequence_mask(self.lengths_P, name='mask')
            # losses = tf.boolean_mask(losses, mask, name='masked_loss')
            # loss_mask = tf.sequence_mask(self.sequence_lengths, self.max_sentence_len, name='mask')
            # loss_mask = tf.reshape(loss_mask, (self.batch_size, self.max_sentence_len))
            # self.loss = tf.contrib.seq2seq.sequence_loss(score, self.labels, tf.cast(loss_mask, tf.float32), name='loss')
            self.loss = tf.contrib.seq2seq.sequence_loss(score, self.labels, tf.cast(loss_mask, tf.float32), name='loss')


            # Add the loss value as a scalar to summary.
            loss_summ = tf.summary.scalar('loss', self.loss)
            self.train_summary.append(loss_summ)


        with tf.name_scope('optimizer'):
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


    def train(self, num_epoch):
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
                for iteration, batch in tqdm(enumerate(generate_batch(self.options.batch_size))):
                    step += 1
                    run_metadata = tf.RunMetadata()
                    feed_dict_train = {}
                    loss, summary = session.run([],
                                                feed_dict=feed_dict_train,
                                                run_metadata=run_metadata)
                    # Add returned summaries to writer in each step, where
                    self.writer.add_summary(summary, step)

                    if step % self.options.step_check == 0:
                        print("RunID:", self.ID)
                        print("Epoch:", epoch, "Iteration:", iteration, "Global Iteration:", step)
                        print("Loss: ", loss)
                        print('--------------------------------')
                        if step != 0:
                            # give back control to callee to run evaluation
                            yield session, step, epoch
                            print('ith-Epoch: {}/{}'.format(epoch, num_epoch))
            self.writer.close()

            # TODO uncomment once the model is implemented
            # Save the model for checkpoints
            # path = self.saver.save(session, os.path.join(tensorboard_dir, 'model.ckpt'))
