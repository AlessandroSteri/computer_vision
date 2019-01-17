import os
import tensorflow as tf
from tqdm import tqdm
from gte.preprocessing.batch import generate_batch, iteration_per_epoch
from gte.info import TB_DIR


class GroundedTextualEntailmentModel(object):
    """Model for Grounded Textual Entailment."""
    def __init__(self, options, ID):
        self.options = options
        self.graph = self.create_graph()
        self.ID = ID

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
            self.other_layer()
            #...
            self.other_layer()
            #...
            self.other_layer()

            # TODO uncomment the saver once the model is implemented
            # creates the saver
            # self.saver = tf.train.Saver()
        print('Created graph!')
        return graph

    def input_layer(self):
        # Define input data tensors.
        with tf.name_scope('inputs'):
            self.placeholder = tf.placeholder(tf.int32, shape=[None, None], name='in')  # shape []


    def other_layer(self):
        pass

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
