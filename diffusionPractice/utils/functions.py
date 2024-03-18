import math

def exists(x):
    '''
    the function that check the parameter is exist.

    Inputs:
        x ( ): input.
    
    Outputs:
        x ( ): True / False
    '''
    return x is not None

def default(val, d):
    '''
    choose the default function.

    Inputs:
        val ( ):
        d ( ): 
    
    Outputs:
        val / d
    '''
    if exists(val):
        return val
    return d() if callable(d) else d

def cast_tuple(t, length = 1):
    '''
    check the 't' is the type of 'tuple'.
    if the 't' wasn't tuple, return the tuple with t for length.

    for example, t is 'a' and lenght is 3, then return ('a', 'a', 'a')
    '''
    if isinstance(t, tuple):
        return t
    return ((t,) * length)

def divisible_by(numer, denom):
    '''
    return (numer % denom) == 0
    '''
    return (numer % denom) == 0

def identity(t, *args, **kwargs):
    '''
    return t
    '''
    return t

def cycle(dl):
    '''
    Inputs:
        dl (list): [data1, data2, data3, ...]
    '''
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    '''
    return (math.sqrt(num) ** 2) == num
    '''
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    '''
    Example:
        Input: num = 10, divisor = 3.
        Ouputs: [3, 3, 3, 1].

    Inputs:
        num (int): total nums.
        divisor (int): each num for group.
    
    Outputs:
        arr (arr): devided arr.
    '''
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    '''
    if image type is same with img_type, return image.
    if image type is different with img_type, convert image type with 'img_type'.
    '''
    if image.mode != img_type:
        return image.convert(img_type)
    return image
