import cv2
import numpy as np
import math
import torch
from collections import defaultdict


from src.vietocr.tool.translate import build_model, translate


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
        s = translate(img, self.model)[0].tolist()

        s = self.vocab.decode(s)

        return s

    def batch_predict(self, images):
        """
        param: images : list of ndarray
        """
        batch_dict, indices = self.batch_process(images)
        list_keys = [i for i in batch_dict if batch_dict[i] != batch_dict.default_factory()]
        result = list([])

        for width in list_keys:
            batch = batch_dict[width]
            batch = np.asarray(batch)
            batch = torch.FloatTensor(batch)
            batch = batch.to(self.config['device'])
            sent = translate(batch, self.model).tolist()

            batch_text = self.vocab.batch_decode(sent)
            result.extend(batch_text)

        # sort text result to original coordinate
        z = zip(result, indices)
        sorted_result = sorted(z, key=lambda element: element[1])
        result, _ = zip(*sorted_result)

        return result

    def preprocess_input(self, image):
        """
        param: image: ndarray of image
        """
        h, w, _ = image.shape
        new_w, image_height = self.resize_v1(w, h, self.config['dataset']['image_height'],
                                             self.config['dataset']['image_min_width'],
                                             self.config['dataset']['image_max_width'])

        img = cv2.resize(image, (new_w, image_height))
        img = np.transpose(img, (2, 0, 1))
        img = img / 255.0

        return img

    def batch_process(self, images):
        batch_img_dict = defaultdict(list)
        image_height = self.config['dataset']['image_height']

        batch_img_li = [self.preprocess_input(img) for img in images]
        batch_imgs, width_list, indices = self.sort_width(batch_img_li, reverse=False)

        min_bucket_width = min(width_list)
        max_width = max(width_list)
        thresh = 30
        max_bucket_width = np.minimum(min_bucket_width + thresh, max_width)

        for i, image in enumerate(batch_imgs):
            w = image.shape[2]

            # reset min_bucket_width, max_bucket_width
            if w > max_bucket_width:
                min_bucket_width = w
                max_bucket_width = np.minimum(min_bucket_width + thresh, max_width)

            avg_bucket_width = int((max_bucket_width + min_bucket_width) / 2)

            new_img = self.resize_v2(image, avg_bucket_width, height=image_height)
            batch_img_dict[avg_bucket_width].append(new_img)

        return batch_img_dict, indices

    @staticmethod
    def sort_width(batch_img, reverse=False):
        def get_img_width(element):
            img = element[0]
            c, h, w = img.shape
            return w

        batch = list(zip(batch_img, range(len(batch_img))))
        sorted_batch = sorted(batch, key=get_img_width, reverse=reverse)
        sorted_batch_img, indices = list(zip(*sorted_batch))

        return sorted_batch_img, list(map(get_img_width, batch)), indices

    @staticmethod
    def resize_v1(w, h, expected_height, image_min_width, image_max_width):
        new_w = int(expected_height * float(w) / float(h))
        round_to = 10
        new_w = math.ceil(new_w / round_to) * round_to
        new_w = max(new_w, image_min_width)
        new_w = min(new_w, image_max_width)

        return new_w, expected_height

    @staticmethod
    def resize_v2(img, width, height):
        new_img = np.transpose(img, (1, 2, 0))
        new_img = cv2.resize(new_img, (width, height), cv2.INTER_AREA)
        new_img = np.transpose(new_img, (2, 0, 1))

        return new_img
