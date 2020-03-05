#!/usr/bin/env python3
"""
Module used to
"""

import numpy as np


def f1_score(confusion):
    """
    calculates the specificity for each class in a confusion matrix:
    Args:
        - confusion: a confusion numpy.ndarray of shape (classes, classes)
        where row indices represent the correct labels and column indices
        represent the predicted labels
            - classes is the number of classes
    Returns: a numpy.ndarray of shape (classes,) with the F1 score of e/class
    Note:
    You may use sensitivity = __import__('1-sensitivity').sensitivity and
                  precision = __import__('2-precision').precision
    """

    # FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
    # FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
    # TP = np.diag(confusion_matrix)
    # TN = confusion_matrix.values.sum() - (FP + FN + TP)

    # 1
    # Sensitivity, hit rate, recall, or true positive rate
    # TPR = TP/(TP+FN)

    # 2
    # Specificity or true negative rate
    # TNR = TN/(TN+FP)

    # 3
    # Precision or positive predictive value
    # PPV = TP/(TP+FP)

    # 4
    # Negative predictive value
    # NPV = TN/(TN+FN)

    # 5
    # Fall out or false positive rate
    # FPR = FP/(FP+TN)

    # 6
    # False negative rate
    # FNR = FN/(TP+FN)

    # 7
    # False discovery rate
    # FDR = FP/(TP+FP)

    # 8
    # Overall accuracy
    # ACC = (TP+TN)/(TP+FP+FN+TN)

    FP = confusion.sum(axis=0) - np.diag(confusion)
    FN = confusion.sum(axis=1) - np.diag(confusion)
    TP = np.diag(confusion)
    TN = confusion.sum() - (FP + FN + TP)

    # 1
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)

    # 3
    # Precision or positive predictive value
    PPV = TP / (TP + FP)

    # F1 = 2 / ((1 / TRP) + (1 / PPV))
    return (2 / ((1 / TPR) + (1 / PPV)))
