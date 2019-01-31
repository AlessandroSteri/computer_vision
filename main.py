import argparse
from gte.model import GroundedTextualEntailmentModel
from gte.utils.log import id_gen
from gte.info import MAX_LEN_P, MAX_LEN_H


def main():
    ### CLI args ###{{{
    cmdLineParser = argparse.ArgumentParser()
    cmdLineParser.add_argument("epoch", type=int, help="Number of full training-set iterations.")
    cmdLineParser.add_argument("batch_size", type=int, help="Number of samples per batch.")
    cmdLineParser.add_argument("learning_rate", type=float, help="Base learning rate.")
    cmdLineParser.add_argument("step_check", type=int, help="Every how many iteration check accuracy over dev sets.")
    cmdLineParser.add_argument("--gpu", dest="use_gpu", action='store_true', help="Enable gpu accelaration.")
    cmdLineParser.add_argument("--dev_env", dest="dev_env", action='store_true', help="Development Environment: Use always the same seed, experiments are repeatable.")
    cmdLineParser.add_argument('--max_len_p', action="store", dest="max_len_p", default=MAX_LEN_P, type=int, help="Max lenght for premises.")
    cmdLineParser.add_argument('--max_len_h', action="store", dest="max_len_h", default=MAX_LEN_H, type=int, help="Max lenght for hypothesis.")
    cmdLineParser.add_argument("--trainable", dest="trainable", action='store_true', help="Makes trainable the pre-trained embeddigs.")
    options = cmdLineParser.parse_args()

    env = ''
    if options.dev_env: env = 'DEV '

    model_info = '' #to log wich layer of the model are used
    exe_id = id_gen() # sort-of unique and monotonic id for tensorboard and logging
    ID = exe_id + ' [' + env +  '_b' + str(options.batch_size) + '_e' + str(options.epoch) + '_l' + str(options.learning_rate) + '_c' + str(options.step_check) + '_model' + '--{}]'.format(model_info)


    gte_model = GroundedTextualEntailmentModel(options, ID)
    for session, step, epoch in gte_model.train(options.epoch):
        #eval
        pass

if __name__ == '__main__':
    main()
