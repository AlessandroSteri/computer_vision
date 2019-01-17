import os


# check if directory exists, if not it makes it
def mkdir(directory, log=False):
    if not exists(directory):
        os.makedirs(directory)
        if log:
            print('Created new directory: {}'.format(directory))

def exists(path):
    return os.path.exists(path)
