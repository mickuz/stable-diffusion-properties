import json
import scipy
import numpy as np

from math import sqrt
from itertools import product

from src.masks import IoUCalculator


MODES = ["raw", "finetuned"]


class ResultRetriever:
    def __init__(self, iou_calculator):
        self.iou_calculator = iou_calculator

        self.results = {}
        self.stats = {}

    def compute_results(self, words, number_of_samples):
        for word, mode in product(words, MODES):
            label = "_".join([word, mode])

            all_iou = self.iou_calculator.get_all_iou(number_of_samples, word, mode)
            mean_iou = IoUCalculator.mean_iou(all_iou)
            median_iou = IoUCalculator.median_iou(all_iou)

            self.results[label] = {
                "iou_scores": all_iou,
                "mean_iou": mean_iou,
                "median_iou": median_iou
            }

    def compute_stats(self, words):
        if not self.results:
            raise ValueError("The results dictionary is empty!")

        for word in words:
            raw_scores = self.results[f"{word}_raw"]["iou_scores"]
            finetuned_scores = self.results[f"{word}_finetuned"]["iou_scores"]

            U1, pvalue1 = scipy.stats.mannwhitneyu(finetuned_scores, raw_scores, alternative="two-sided")
            U2, pvalue2 = scipy.stats.mannwhitneyu(raw_scores, finetuned_scores, alternative="two-sided")
            d = ResultRetriever.cohen_d(finetuned_scores, raw_scores)

            self.stats[word] = {
                "test_stat_1": U1,
                "p_value_1": pvalue1,
                "test_stat_2": U2,
                "p_value_2": pvalue2,
                "cohen_d": d
            }

    def save_results(self, results_path):
        with open(results_path, mode="w") as file:
            json.dump(self.results, file)

    def save_stats(self, stats_path):
        with open(stats_path, mode="w") as file:
            json.dump(self.stats, file)
    
    @staticmethod
    def cohen_d(d1, d2):
        n1, n2 = len(d1), len(d2)
        s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
        s = sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
        u1, u2 = np.mean(d1), np.mean(d2)

        return abs(u1 - u2) / s
    