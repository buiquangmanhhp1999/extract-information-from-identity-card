from vietocr.tool.translate import build_model, translate, translate_beam_search, process_input, predict
from vietocr.tool.utils import download_weights
import cv2
import numpy as np
import math

import torch


class Predictor(object):
    def __init__(self, config):
        device = config['device']

        model, vocab = build_model(config)
        weights = config['weights']

        model.load_state_dict(torch.load(weights, map_location=torch.device(device)))

        self.config = config
        self.model = model
        self.vocab = vocab

    def predict(self, img):
        img = self.preprocess_input(img)
        img = np.expand_dims(img, axis=0)
        img = torch.FloatTensor(img)
        img = img.to(self.config['device'])

        if self.config['predictor']['beamsearch']:
            sent = translate_beam_search(img, self.model)
            s = sent
        else:
            s = translate(img, self.model)[0].tolist()

        s = self.vocab.decode(s)

        return s

    def batch_predict(self, images):
        """
        param: images : list of ndarray
        """
        batch = self.batch_process(images)
        batch = batch.to(self.config['device'])
        if self.config['predictor']['beamsearch']:
            sent = translate_beam_search(batch, self.model)
            s = sent
        else:
            sents = translate(batch, self.model).tolist()

        sequences = self.vocab.batch_decode(sents)
        return sequences

    def preprocess_input(self, image):
        """
        param: image: ndarray of image
        """
        h, w, _ = image.shape
        new_w, image_height = self.resize(w, h, self.config['dataset']['image_height'],
                                          self.config['dataset']['image_min_width'],
                                          self.config['dataset']['image_max_width'])

        img = cv2.resize(image, (new_w, image_height))
        img = np.transpose(img, (2, 0, 1))
        img = img / 255
        return img

    def batch_process(self, images):
        batch = []
        for image in images:
            img = self.preprocess_input(image)
            c, h, w = img.shape
            if w < 160:
                zeros_img = np.zeros((3, 32, 160))
                zeros_img[:, :, :w] = img
                img = zeros_img
            elif w > 160:
                img = np.transpose(img, (1, 2, 0))
                img = cv2.resize(img, (160, 32), cv2.INTER_AREA)
                img = np.transpose(img, (2, 0, 1))
            batch.append(img)
        batch = np.asarray(batch)
        batch = torch.FloatTensor(batch)
        return batch

    @staticmethod
    def resize(w, h, expected_height, image_min_width, image_max_width):
        new_w = int(expected_height * float(w) / float(h))
        round_to = 10
        new_w = math.ceil(new_w / round_to) * round_to
        new_w = max(new_w, image_min_width)
        new_w = min(new_w, image_max_width)

        return new_w, expected_height
