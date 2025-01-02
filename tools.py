"""
Utility Functions for Speaker Verification
Originally adapted from voxceleb_trainer:
https://github.com/clovaai/voxceleb_trainer/blob/master/tuneThreshold.py
"""

import os
import numpy as np
import torch

from operator import itemgetter
from sklearn import metrics
import torch.nn.functional as F


def init_args(args):
    """
    Initialize file paths for saving scores and model checkpoints.

    Args:
        args: Command-line or config arguments object, assumed to have 
              'save_path' attribute.

    Returns:
        Updated args with new attributes:
          - score_save_path: The path to save score logs.
          - model_save_path: The path to save model checkpoints.
    """
    args.score_save_path = os.path.join(args.save_path, 'score.txt')
    args.model_save_path = os.path.join(args.save_path, 'model')
    os.makedirs(args.model_save_path, exist_ok=True)
    return args


def tuneThresholdfromScore(scores, labels, target_fa, target_fr=None):
    """
    Compute threshold(s) based on desired false-accept (FA) or false-reject (FR) rates.
    
    This function uses an ROC curve to derive specific thresholds that match
    given FA or FR targets. It also computes the EER (Equal Error Rate).

    Args:
        scores (list or np.array): List/array of similarity scores from pairs/trials.
        labels (list or np.array): Binary labels (1 for same speaker, 0 for different).
        target_fa (list of floats): Desired false acceptance rates.
        target_fr (list of floats, optional): Desired false reject rates.

    Returns:
        tunedThreshold (list): A list of [threshold, false_accept_rate, false_reject_rate].
        eer (float): Equal Error Rate (percentage).
        fpr (np.array): False-positive rates at each threshold.
        fnr (np.array): False-negative rates at each threshold.
    """
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    tunedThreshold = []

    # If target_fr is provided, find thresholds closest to each FR target.
    if target_fr:
        for tfr in target_fr:
            idx = np.nanargmin(np.absolute((tfr - fnr)))
            tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])

    # Find thresholds closest to each FA target.
    for tfa in target_fa:
        idx = np.nanargmin(np.absolute((tfa - fpr)))
        tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])

    # EER is the rate where FPR and FNR are equal (or closest to each other).
    idxE = np.nanargmin(np.absolute((fnr - fpr)))
    eer = max(fpr[idxE], fnr[idxE]) * 100

    return tunedThreshold, eer, fpr, fnr


def ComputeErrorRates(scores, labels):
    """
    Create lists of false-negative rates (FNR), false-positive rates (FPR),
    and thresholds. These can be used for plotting DET curves or computing
    additional metrics like minDCF.

    Args:
        scores (list or np.array): Verification scores.
        labels (list or np.array): Binary labels.

    Returns:
        fnrs (list of floats): False-negative rates at each threshold.
        fprs (list of floats): False-positive rates at each threshold.
        thresholds (list of floats): Threshold values at which FNR and FPR are computed.
    """
    # Sort scores (and keep track of their indices)
    sorted_indexes, thresholds = zip(*sorted(
        [(index, threshold) for index, threshold in enumerate(scores)],
        key=itemgetter(1)
    ))

    # Rearrange labels to match sorted scores
    labels_sorted = [labels[i] for i in sorted_indexes]

    fnrs = []
    fprs = []

    # Compute cumulative false negatives and false positives
    for i, lab in enumerate(labels_sorted):
        if i == 0:
            fnrs.append(lab)
            fprs.append(1 - lab)
        else:
            fnrs.append(fnrs[i - 1] + lab)
            fprs.append(fprs[i - 1] + 1 - lab)

    # Normalize FNR and FPR
    total_positives = sum(labels_sorted)  # total positives
    total_negatives = len(labels_sorted) - total_positives  # total negatives

    fnrs = [x / float(total_positives) for x in fnrs] if total_positives > 0 else fnrs
    fprs = [1 - (x / float(total_negatives)) for x in fprs] if total_negatives > 0 else fprs

    return fnrs, fprs, thresholds


def ComputeMinDcf(fnrs, fprs, thresholds, p_target, c_miss, c_fa):
    """
    Computes the minimum detection cost function (minDCF) over a range of thresholds.

    This is a standard metric for speaker verification (and other detection tasks),
    as defined by the NIST evaluations.

    Args:
        fnrs (list of floats): False-negative rates.
        fprs (list of floats): False-positive rates.
        thresholds (list of floats): Corresponding thresholds for fnrs/fprs.
        p_target (float): Prior probability of the target condition (same speaker).
        c_miss (float): Cost of a miss (false reject).
        c_fa (float): Cost of a false accept.

    Returns:
        min_dcf (float): The minimum detection cost function value.
        min_c_det_threshold (float): Threshold at which minDCF is attained.
    """
    min_c_det = float("inf")
    min_c_det_threshold = thresholds[0]

    for i in range(len(fnrs)):
        # Weighted sum of false negative and false positive rates
        c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]

    # Normalization to get the actual minDCF
    # c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def

    return min_dcf, min_c_det_threshold


def accuracy(output, target, topk=(1,)):
    """
    Compute top-k accuracy for classification outputs.

    Args:
        output (torch.Tensor): Logits or scores of shape (batch_size, num_classes).
        target (torch.Tensor): Class labels of shape (batch_size,).
        topk (tuple): Tuple of k values for which to compute accuracy.

    Returns:
        list[torch.Tensor]: A list of accuracies for each top-k value in percentage.
                            E.g., [top1_acc, top5_acc] if topk=(1,5).
    """
    maxk = max(topk)
    batch_size = target.size(0)

    # Get the indices of the top-k predictions
    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()

    # Compare predictions with the ground-truth labels
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    results = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        results.append(correct_k.mul_(100.0 / batch_size))

    return results