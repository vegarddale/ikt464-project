import numpy as np

def calculate_iou(y_true, y_pred, num_classes):
    iou_scores = []

    for class_id in range(num_classes):
        true_positive = np.sum((y_true == class_id) & (y_pred == class_id))
        false_positive = np.sum((y_true != class_id) & (y_pred == class_id))
        false_negative = np.sum((y_true == class_id) & (y_pred != class_id))

        intersection = true_positive
        union = true_positive + false_positive + false_negative

        iou = intersection / union if union != 0 else 0
        iou_scores.append(iou)

    return iou_scores