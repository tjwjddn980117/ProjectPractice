import os
import errno

def mkdir_p(path):
    '''
    The function that make dirs.

    Inputs:
        path (str): the path we have to make.
    '''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise