from PIL import Image
import yaml

from src.vietocr.tool.predictor import Predictor
from src.config import text_recognition


class TextRecognition(object):
    def __init__(self):
        self.ocr_config = self.load_config()
        self.detector = Predictor(self.ocr_config)

    def load_config(self):
        # load base config
        ocr_config = self.read_from_config(file_yml=text_recognition['base_config'])

        # load vgg transformer config
        vgg_config = self.read_from_config(file_yml=text_recognition['vgg_config'])

        # update base config
        ocr_config.update(vgg_config)

        # load model from checkpoint
        ocr_config['weights'] = text_recognition['model_weight']
        ocr_config['predictor']['beamsearch'] = False

        return ocr_config

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
