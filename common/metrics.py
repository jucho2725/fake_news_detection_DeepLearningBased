"""
Simple implementation of evaluation metrics.

last update : Nov 27th, 2019
"""

def accuracy(pred, target):
    pred_y = pred >= 0.5
    num_correct = target.eq(pred_y.float()).sum()
    accuracy = (num_correct.item() * 100.0 / len(target))
    return accuracy

def f1_measure(pred, target):
    tp = sum([p == 1 and t == 1 for p, t in zip(pred, target)])
    tn = sum([p == 0 and t == 0 for p, t in zip(pred, target)])
    fp = sum([p == 1 and t == 0 for p, t in zip(pred, target)])
    fn = sum([p == 0 and t == 1 for p, t in zip(pred, target)])

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score