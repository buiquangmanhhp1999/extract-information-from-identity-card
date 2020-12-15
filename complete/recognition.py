from PIL import Image
import os
import gdown
import yaml
from complete.predictor import Predictor


class TextRecognition(object):
    def __init__(self, path_to_checkpoint):
        self.config = self.load_config(path_to_checkpoint)
        self.detector = Predictor(self.config)

    def load_config(self, path_to_checkpoint):
        url_base = '1xiw7ZnT3WH_9HXoGpLbhW-m2Sm2nlthi'
        url_config_vgg_transformers = '1TF8effeufpgkHqQFlmNWKsQtCMfDiooa'

        # load base config
        if os.path.isfile('./config_text_recognition/base.yml'):
            base_config = self.read_from_config(file_yml='./config_text_recognition/base.yml')
        else:
            base_config = self.download_config(url_base)

        # load vgg transformer config
        if os.path.isfile('./config_text_recognition/vgg-transformer.yml'):
            config = self.read_from_config(file_yml='./config_text_recognition/vgg-transformer.yml')
        else:
            config = self.download_config(url_config_vgg_transformers)

        # update base config
        base_config.update(config)

        # load model from checkpoint
        base_config['weights'] = path_to_checkpoint
        base_config['device'] = 'cpu'
        base_config['predictor']['beamsearch'] = False

        return base_config

    @staticmethod
    def download_config(url_id):
        url = 'https://drive.google.com/uc?id={}'.format(url_id)
        output = gdown.download(url, quiet=True)

        with open(output, encoding='utf-8') as f:
            config = yaml.safe_load(f)

        return config

    @staticmethod
    def read_from_config(file_yml):
        with open(file_yml, encoding='utf-8') as f:
            config = yaml.safe_load(f)

        return config

    def predict(self, image):
        image = Image.fromarray(image)
        result = self.detector.predict(image)

        return result

    def predict_on_batch(self, batch_images):
        return self.detector.batch_predict(batch_images)
