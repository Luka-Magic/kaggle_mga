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
        indices = np.column_stack(np.unravel_index(
            flat_indices, output.shape))[-n_points:, :]  # (n_points, 2)
        output_set = set((x, y) for y, x in indices)

        n_bs_corrects += len(target_set & output_set)
        n_bs_points += n_points

    return n_bs_corrects / n_bs_points, n_bs_points


def is_nan(value):
    """
    Check if a value is NaN (not a number).

    Args:
        value (int, float, str): The value to check

    Returns:
        bool: True if the value is NaN, False otherwise
    """
    return isinstance(value, float) and np.isnan(value)


def tensor2arr(tensor_img):
    if len(tensor_img.shape) == 2:
        tensor_img = tensor_img.unsqueeze(0)
    return tensor_img.permute(1, 2, 0).numpy() * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])


class PointCounter():
    def __init__(self, cfg):
        k_s = 5
        kernel = np.zeros((k_s, k_s))
        c = k_s // 2
        for x in range(k_s):
            for y in range(k_s):
                kernel[x][y] = np.exp(- ((x - c) ** 2 +
                                      (y - c) ** 2) / (2 * cfg.sigma ** 2))
        self.kernel = kernel
        self.kernel_size = k_s

    def _calc_kl_divergence(self, pred_kernel, gt_kernel):
        p = np.clip(pred_kernel.flatten(), 1e-7, 1.0)
        p /= np.sum(p)
        q = np.clip(gt_kernel.flatten(), 1e-7, 1.0)
        q /= np.sum(q)
        kl_div = np.sum(np.where(p != 0, p * np.log(p / q), 0))
        return kl_div

    def count(self, y_pred, thrs, wandb_thr=None):
        '''
            y_pred: torch.tensor (bs, 1, hm_h, hm_w)
        '''

        bs, _, h, w = y_pred.shape
        y_pred = torch.sigmoid(y_pred).squeeze(1).detach().cpu().numpy()
        pad = self.kernel_size // 2
        y_pred_pad = np.pad(y_pred, ((0, 0), (pad, pad), (pad, pad)))

        n_counts_list = []
        pre_arr = np.ones_like(y_pred) * 100

        if not isinstance(thrs, list):
            thrs = [thrs]

        n_count_dict = {thr: [] for thr in thrs}
        for i in range(bs):
            counter_per_data = {thr: 0 for thr in thrs}
            for y in range(pad, h+pad):
                for x in range(pad, w+pad):
                    # (x, y)が中心の3*3のカーネルで、中心の値が一番大きい時のみ通過
                    if not np.argmax(y_pred_pad[i, y-1:y+2, x-1:x+2]) == 4:
                        score = 100.0
                    else:
                        pred_kernel = y_pred_pad[i,
                                                 y-pad:y+pad+1, x-pad:x+pad+1]
                        score = self._calc_kl_divergence(
                            pred_kernel, self.kernel)

                    pre_arr[i, y-2, x-2] = score
                    for thr in thrs:
                        if score < thr:
                            counter_per_data[thr] += 1
            for thr in thrs:
                n_count_dict[thr].append(counter_per_data[thr])

        return n_count_dict, pre_arr
