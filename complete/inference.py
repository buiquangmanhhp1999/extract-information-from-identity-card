import cv2
from detector import Detector
from inference_tflite import DetectorFaster
import argparse
import time


def get_args():
    arg = argparse.ArgumentParser()
    arg.add_argument('-i', '--image_path', help='link to image', default='./img_10.png')
    arg.add_argument('-o', '--option', help='detection or detection_faster', default='detection_faster')
    return arg.parse_args()


