import os
import logging

import numpy as np

from PIL import Image
from transformers import pipeline

from torchvision.transforms.functional import pil_to_tensor

from daam.experiment import GenerationExperiment
from daam.heatmap import GlobalHeatMap
from daam.evaluate import compute_iou


class IoUCalculator:
    def __init__(self, output_dir, segmentation_model):
      self.output_dir = output_dir
      self.segmentation_model = segmentation_model

    def get_ground_truth_mask(self, word, idx, mode="raw"):
        experiment_name = f"{mode}-photo-of-a-{word}-{idx}"
        image_name = "output.png"
        image_path = os.path.join(self.output_dir, mode, experiment_name, image_name)

        image = Image.open(image_path)
        segmentation_pipeline = pipeline("image-segmentation", self.segmentation_model)
        results = segmentation_pipeline(image)

        for category_dict in results:
            if not all(key in category_dict for key in ("label", "mask")):
                continue

            if category_dict["label"] == word:
                image_gt_mask = category_dict["mask"]

        try:
            ground_truth_mask = pil_to_tensor(image_gt_mask).squeeze(0)
        except (UnboundLocalError, ValueError) as err:
            logging.error(err)
            return None

        return ground_truth_mask

    def get_prediction_mask(self, word, idx, mode="raw"):
        experiment_name = experiment_name = f"{mode}-photo-of-a-{word}-{idx}"
        experiment_path = os.path.join(self.output_dir, mode, experiment_name)

        experiment = GenerationExperiment.load(experiment_path)
        heat_map = GlobalHeatMap(experiment.tokenizer, experiment.prompt, experiment.global_heat_map)

        try:
            prediction_mask = heat_map.compute_word_heat_map(word, word_idx=3).expand_as(experiment.image, threshold=0.4, plot=False)
        except (IndexError, ValueError) as err:
            logging.error(err)
            return None

        return prediction_mask

    def get_all_iou(self, number_of_samples, word, mode="raw"):
        all_iou = []

        for idx in range(number_of_samples):
            ground_truth_mask = self.get_ground_truth_mask(word, idx, mode)
            prediction_mask = self.get_prediction_mask(word, idx, mode)
            iou = IoUCalculator.get_single_iou(ground_truth_mask, prediction_mask)
            all_iou.append(iou)

        return all_iou

    @staticmethod
    def get_single_iou(ground_truth_mask, prediction_mask):
        if ground_truth_mask is None or prediction_mask is None:
            return 0.0

        normalized_ground_truth_mask = (ground_truth_mask / 255).int()
        normalized_prediction_mask = prediction_mask.int()

        iou = compute_iou(normalized_ground_truth_mask, normalized_prediction_mask)

        return iou

    @staticmethod
    def mean_iou(all_iou):
        return np.mean(all_iou)

    @staticmethod
    def median_iou(all_iou):
        return np.median(all_iou)