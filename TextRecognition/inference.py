from PIL import Image
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import cv2
import time

config = Cfg.load_config_from_name('vgg_transformer')
config['export'] = 'transformerocr_checkpoint.pth'
config['device'] = 'cpu'
config['predictor']['beamsearch'] = False

start1 = time.time()

detector = Predictor(config)
end1 = time.time()

img = Image.open('./ANH_1321.jpeg')
print("Load image: ", end1 - start1)
start = time.time()
print(detector.predict(img))
end = time.time()
print('Required time: ', end - start)
cv2.imshow('image', np.array(img))
cv2.waitKey(0)
