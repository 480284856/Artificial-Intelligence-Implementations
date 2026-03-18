import numpy as np

"""
METRICS MODULE
--------------
This script serves as the evaluation 'Scoreboard' for the neural network. 
It provides several ways to measure how well the model recognizes gestures, 
moving beyond simple accuracy to handle imbalanced datasets (e.g., many 'Idle' 
samples vs. fewer 'Gesture' samples).

Core Functionality:
1. Foundation: The Confusion Matrix acts as the raw data source for all 
   advanced metrics.
2. Fairness: Balanced Accuracy and F1-Score ensure that rare gestures 
   are treated with the same importance as frequent 'Idle' states.
3. Precision: IoU (Intersection over Union) measures the specific overlap 
   between predicted and actual class labels.
"""

def accuracy(y_true, y_pred):
    """
    Calculates the overall accuracy of the model.
    Accuracy is the ratio of correctly predicted observations to the total observations.
    """
    return float(np.mean(y_true == y_pred))

def confusion_matrix(y_true, y_pred, num_classes):
    """
    Computes a confusion matrix to evaluate classification accuracy.
    Rows represent the ground truth (actual labels), while columns represent 
    the model's predictions.
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        cm[int(t), int(p)] += 1
    return cm

def f1_score_from_cm(cm, eps=1e-12):
    """
    Calculates the mean per-class F1-Score from a confusion matrix.
    The F1-score is the harmonic mean of precision and recall. 
    
    -> We distinguish between Precision and Recall to understand the model's behavior. 
    A model with high Precision acts as a perfectionist—it only triggers when it is absolutely sure, 
    prioritizing the reliability of its claims. In contrast, a model with high Recall acts as an 
    over-eager catcher—it is 'restless' and tries to detect every movement to maximize its hit rate, 
    even at the cost of being less picky.

    This metric is particularly useful for imbalanced datasets.
    """
    # True Positives (diagonal), False Positives (column sum - TP), False Negatives (row sum - TP)
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0).astype(np.float64) - tp
    fn = cm.sum(axis=1).astype(np.float64) - tp
    
    # Calculate Precision and Recall per class
    precision = tp / np.maximum(tp + fp, eps)
    recall = tp / np.maximum(tp + fn, eps)
    
    # Calculate F1 per class and return the mean
    f1 = 2 * (precision * recall) / np.maximum(precision + recall, eps)
    return float(np.mean(f1))

def iou_from_cm(cm, eps=1e-12):
    """
    Calculates the Intersection over Union (IoU) per class.
    IoU measures the overlap between the predicted and actual labels
    """
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0).astype(np.float64) - tp
    fn = cm.sum(axis=1).astype(np.float64) - tp
    
    denominator = tp + fp + fn
    return (tp / np.maximum(denominator, eps)).astype(np.float64)

def mean_iou(iou_array, include=None):
    """
    Returns the average IoU across all or specific classes.
    'include' can be a list of indices (e.g., [0, 1, 2] to ignore 'idle' at index 3).
    """
    if include is None:
        return float(np.mean(iou_array))
    if len(include) == 0:
        return float("nan")
    return float(np.mean(iou_array[np.array(include, dtype=np.int64)]))

def balanced_accuracy_from_cm(cm, eps=1e-12):
    """
    Mean per-class recall (a.k.a. balanced accuracy).
    Useful when classes are imbalanced.
    """
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)
    recall = tp / np.maximum(support, eps)
    return float(np.mean(recall))