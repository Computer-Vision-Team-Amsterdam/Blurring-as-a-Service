import numpy as np


class TotalBlurredArea:
    def __init__(self):
        self.f1_score = None
        self.recall = None
        self.precision = None
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

    def update_statistics_based_on_masks(self, true_mask, predicted_mask):
        """
        Computes statistics for a given pair of binary masks.
        Args:
            true_mask: numpy array of shape (height, width)
            predicted_mask: numpy array of shape (height, width)

        Returns:

        """ """
        Computes true positives and false positives for a given pair of binary masks.
        :param true_mask: numpy array of shape (height, width)
        :param pred_mask: numpy array of shape (height, width)
        """
        tp = np.sum((true_mask == 1) & (predicted_mask == 1))
        fp = np.sum((true_mask == 0) & (predicted_mask == 1))
        tn = np.sum((true_mask == 0) & (predicted_mask == 0))
        fn = np.sum((true_mask == 1) & (predicted_mask == 0))

        self.tp += tp
        self.fp += fp
        self.tn += tn
        self.fn += fn

    def get_statistics(self):
        self.precision = (
            round(self.tp / (self.tp + self.fp), 3) if self.tp + self.fp > 0 else None
        )
        self.recall = (
            round(self.tp / (self.tp + self.fn), 3) if self.tp + self.fn > 0 else None
        )
        self.f1_score = round(
            2 * self.precision * self.recall / (self.precision + self.recall), 3
        )

        return {
            "true_positives": self.tp,
            "false_positives": self.fp,
            "true_negatives": self.tn,
            "false_negatives:": self.fn,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
        }
