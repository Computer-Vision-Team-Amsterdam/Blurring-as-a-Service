import numpy as np


class TotalBlurredArea:
    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

    def add_mask(self, true_mask, pred_mask):
        """
        Computes true positives and false positives for a given pair of binary masks.
        :param true_mask: numpy array of shape (height, width)
        :param pred_mask: numpy array of shape (height, width)
        """
        tp = np.sum((true_mask == 1) & (pred_mask == 1))
        fp = np.sum((true_mask == 0) & (pred_mask == 1))
        tn = np.sum((true_mask == 0) & (pred_mask == 0))
        fn = np.sum((true_mask == 1) & (pred_mask == 0))
        self.tp += tp
        self.fp += fp
        self.tn += tn
        self.fn += fn

    def summary(self, message: str = ""):
        precision = self.tp / (self.tp + self.fp)
        recall = self.tp / (self.tp + self.fn)

        f1 = 2 * precision * recall / (precision + recall)

        print(f"{message}")
        print(f"TP: {self.tp:,}\nFP: {self.fp:,}\nTN: {self.tn:,}\nFN: {self.fn:,}\n")
        print(f"Precision: {precision:.3f}.")
        print(f"Recall: {recall:.3f}.")
        print(f"F1 score: {f1:.3f}.")
