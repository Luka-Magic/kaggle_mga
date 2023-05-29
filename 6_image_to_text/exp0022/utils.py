import os
import numpy as np
import torch
import warnings
from typing import List, Dict, Union, Tuple, Any
import math


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    warnings.simplefilter('ignore')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def round_float(value: Union[int, float, str]) -> Union[str, float]:
    """
    Convert a float value to a string with the specified number of decimal places. 
    If there is more than 1 digit in the integer, then we will truncate to 1 decimal.
    Otherwise, will truncate to 4 decimals.

    Args:
        value (int, float, str): The float value to convert

    Returns:
        str: The rounded float value as a string
    """
    if isinstance(value, float):
        value = str(value)

        if "." in value:
            integer, decimal = value.split(".")
            if abs(float(integer)) > 1:
                decimal = decimal[:1]
            else:
                decimal = decimal[:4]

            value = integer + "." + decimal
    return value


def convert_num_to_2digits(number):
    '''
        123 => 1,2
        584000 => 6,5
        0.003 => 3,-3
        -16.234 => -2,1
        3 => 3,0
    '''
    if number == 0:
        return [0, 0]

    sign = int(math.copysign(1, number))
    magnitude = abs(number)

    digit1 = math.floor(math.log10(magnitude))
    digit2 = round(magnitude / 10**digit1)

    return f'{sign * digit2},{digit1}'


def convert_2digit_to_num(string):
    '''
        123 => 1,2
        584000 => 6,5
        0.003 => 3,-3
        -16.234 => -2,1
        3 => 3,0
    '''
    digit1, digit2 = map(int, string.split(','))
    if digit1 == 0:
        return 0

    sign = int(math.copysign(1, digit1))
    magnitude = abs(digit1) * 10**digit2

    return sign * magnitude


def is_nan(value: Union[int, float, str]) -> bool:
    """
    Check if a value is NaN (not a number).

    Args:
        value (int, float, str): The value to check

    Returns:
        bool: True if the value is NaN, False otherwise
    """
    return isinstance(value, float) and np.isnan(value)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
