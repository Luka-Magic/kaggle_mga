import os
import numpy as np
import torch
import warnings
from typing import List, Dict, Union, Tuple, Any
import math
from metrics import normalized_rmse


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


def is_nan(value: Union[int, float, str]) -> bool:
    """
    Check if a value is NaN (not a number).

    Args:
        value (int, float, str): The value to check

    Returns:
        bool: True if the value is NaN, False otherwise
    """
    return isinstance(value, float) and np.isnan(value)


def reduce_precision(arr):
    for i in range(-7, 7):
        # Round array
        prec = np.round(arr, decimals=i)
        if i <= 0:
            prec = prec.astype(int)
        prec = list(prec)
        # Check if nrmse is close enough
        if normalized_rmse(arr, prec) >= 0.96:
            return prec
    return arr


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
