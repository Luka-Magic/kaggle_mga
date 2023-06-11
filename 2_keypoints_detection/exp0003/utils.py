import os
import numpy as np
import torch
import warnings
import cv2

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


def calc_accuracy(outputs, targets):
    '''
    でかい順に並べてどれだけ一致するか
    target: (bs * h * w)
    '''
    n_bs_corrects = 0
    n_bs_points = 0

    for i, (target, output) in enumerate(zip(targets, outputs)):
        target_ys, target_xs = np.where(target == 1.)
        target_set = set((x, y) for x, y in zip(target_xs, target_ys))
        n_points = len(target_set)

        output = output[0, :, :]
        flat_indices = np.argsort(output.ravel())
        indices = np.column_stack(np.unravel_index(flat_indices, output.shape))[-n_points:, :] # (n_points, 2)
        output_set = set((x, y) for y, x in indices)

        n_bs_corrects += len(target_set & output_set)
        n_bs_points += n_points

    return n_bs_corrects / n_bs_points, n_bs_points







        

