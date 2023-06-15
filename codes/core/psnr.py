"""
Computes PSNR of a batch of monochrome images.
NOTE that a numpy version and torch.Tensor version have slightly different values.
e9b29ba0b21f3b5fbd0f915309dcd18ecfee0f55
Taken from https://github.com/ashesh-0/Disentangle
"""

import numpy as np
import torch


def allow_numpy(func):
    """
    All optional arguements are passed as is. positional arguments are checked. if they are numpy array,
    they are converted to torch Tensor.
    """

    def numpy_wrapper(*args, **kwargs):
        new_args = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                arg = torch.Tensor(arg)
            new_args.append(arg)
        new_args = tuple(new_args)

        output = func(*new_args, **kwargs)
        return output

    return numpy_wrapper


def zero_mean(x):
    return x - torch.mean(x, dim=1, keepdim=True)


def fix_range(gt, x):
    a = torch.sum(gt * x, dim=1, keepdim=True) / (torch.sum(x * x, dim=1, keepdim=True))
    return x * a


def fix(gt, x):
    gt_ = zero_mean(gt)
    return fix_range(gt_, zero_mean(x))


def _PSNR_internal(gt, pred, range_=None):
    if range_ is None:
        range_ = torch.max(gt, dim=1).values - torch.min(gt, dim=1).values

    mse = torch.mean((gt - pred) ** 2, dim=1)
    return 20 * torch.log10(range_ / torch.sqrt(mse))


@allow_numpy
def PSNR(gt, pred, range_=None):
    '''
        Compute PSNR.
        Parameters
        ----------
        gt: array
            Ground truth image.
        pred: array
            Predicted image.
    '''
    assert len(gt.shape) == 3, 'Images must be in shape: (batch,H,W)'

    gt = gt.view(len(gt), -1)
    pred = pred.view(len(gt), -1)
    return _PSNR_internal(gt, pred, range_=range_)


@allow_numpy
def RangeInvariantPsnr(gt, pred):
    """
    NOTE: Works only for grayscale images.
    Adapted from https://github.com/juglab/ScaleInvPSNR/blob/master/psnr.py
    It rescales the prediction to ensure that the prediction has the same range as the ground truth.
    """
    assert len(gt.shape) == 3, 'Images must be in shape: (batch,H,W)'
    gt = gt.view(len(gt), -1)
    pred = pred.view(len(gt), -1)
    ra = (torch.max(gt, dim=1).values - torch.min(gt, dim=1).values) / torch.std(gt, dim=1)
    gt_ = zero_mean(gt) / torch.std(gt, dim=1, keepdim=True)
    return _PSNR_internal(zero_mean(gt_), fix(gt_, pred), ra)
