import numpy as np
from typing import Dict

# How many of predicted positive classes are actually correct
def compute_precision(confusion_matrix: Dict):
    return confusion_matrix['TP'] / (confusion_matrix['TP'] + confusion_matrix['FP'] + 1e-9)


# How many of the actual positive classes are actually correct
def compute_recall(confusion_matrix: Dict):
    return confusion_matrix['TP'] / (confusion_matrix['TP'] + confusion_matrix['FN'] + 1e-9)


def compute_F1(confusion_matrix: Dict):
    precision = compute_precision(confusion_matrix)
    recall = compute_recall(confusion_matrix)

    return 2 * precision * recall / (precision + recall + 1e-9)


def compute_accuracy(confusion_matrix: Dict):
    tp = confusion_matrix['TP']
    fp = confusion_matrix['FP']
    tn = confusion_matrix['TN']
    fn = confusion_matrix['FN']

    return (tp + tn) / (tp + fp + tn + fn + 1e-9)


def compute_pr_curve(predProbs, trueY):
    # Create range of thresholds to sweep across
    thresholds = np.sort(np.unique(predProbs))[::-1]

    precisions = []
    recalls = []
    for threshold in thresholds:
        predY = (predProbs >= threshold).astype(int)
        cm = compute_confusion_matrix(predY, trueY)
        precisions.append(compute_precision(**cm))
        recalls.append(compute_recall(**cm))

    return {'thresholds' : thresholds,
            'precisions' : precisions,
            'recalls' : recalls}


def compute_roc_curve(predProbs, trueY):
    thresholds = np.sort(np.unique(predProbs))[::-1]    # Sort preds in descending order

    tprs = []
    fprs = []
    for threshold in thresholds:
        predY = (predProbs >= threshold).astype(int)
        cm = compute_confusion_matrix(predY, trueY)
        tpr = compute_recall(**cm)
        fpr = cm['FP'] / (cm['FP'] + cm['TN'] + 1e-9)

        tprs.append(tpr)
        fprs.append(fpr)
    
    tprs = np.array([0.0], tprs, [1.0])
    fprs = np.array([0.0], fprs, [1.0])
    
    return {'thresholds' : thresholds,
            'tprs' : tprs,
            'fprs' : fprs}
        

def compute_average_precision(predY, trueY):
    pass


def compute_mean_average_precision(predY, trueY):
    pass


def compute_confusion_matrix(predY, trueY):
    """
    TP : Predicted - positive, groundtruth - positive
    FP : Predicted - positive, groundtruth - negative
    TN : Predicted - negative, groundtruth - negative
    FN : Predicted - negative, groundtruth - positive
    """

    truePositive = np.sum((predY == 1) & (trueY == 1))
    falsePositive = np.sum((predY == 1) & (trueY == 0))
    trueNegative = np.sum((predY == 0) & (trueY == 0))
    falseNegative = np.sum((predY == 0) & (trueY == 1))

    return {'TP' : truePositive,
            'FP' : falsePositive,
            'TN' : trueNegative,
            'FN' : falseNegative}