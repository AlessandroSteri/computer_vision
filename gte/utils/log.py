from time import time
from functools import wraps
import inspect
import uuid
from termcolor import colored

def id_gen(unique=True, sequential=True):
    assert unique or sequential
    seq_id = ''
    unq_id = ''
    if sequential:
        seq_id = str(int(time()*100))
    if unique:
        unq_id = str(uuid.uuid4()).replace('-', '').upper()[0:12]
    return seq_id + '_' + unq_id

#returns percentage approximated to num_decimal both as int and as string
def calculate_percentage(numeartor, denumerator, num_decimals):
    approx = 10**num_decimals
    float_percentage = (numeartor*100) / denumerator
    approx_perc = int(float_percentage * approx) / approx
    return approx_perc, str(approx_perc)+'%'


def log(text_to_log, log_file=None):
    if log_file is None:
        print(text_to_log)
    else:
        with open(log_file, "a") as l:
            l.write(text_to_log)


def log_call(func):
    """ Decorate a function to print its name and arguments.
    """
    @wraps(func)
    def func_with_log(*args, **kwargs):
        name = func.__name__
        str_args = str(args)[1:-1]
        log(colored('[Fn] ', 'green') + _get_func_module() + '.' + str(name)+ '(' +str(str_args)+ ' kwargs: '+str(kwargs) + ')')
        return func(*args, **kwargs)
    return func_with_log


#TODO:se passo file log su file
@log_call
def f(x):
    print('f called')


def _get_func_module():
    frm = inspect.stack()[1]
    mod = inspect.getmodule(frm[0])
    return mod.__name__


class LogTime():
    def __init__(self, log_string, mode="s"):
        self.start = None
        self.log_string = log_string
        self.mode = mode
        self.unit = 1
        if mode == 's':
            self.unit = 60**0
        elif mode == 'm':
            self.unit = 60**1
        elif mode == 'h':
            self.unit = 60**2
        else:
            self.mode = 's'

    def __enter__(self):
        self.start = time()

    def __exit__(self, *args):
        print("[LogTime] {}: {}{}".format(self.log_string, (time() - self.start)/self.unit, self.mode))

#TODO: test file
def main():
    with LogTime('Dumb function elpsed time'):
        for i in range(1000000):
            pass

if __name__ == '__main__':
    main()
