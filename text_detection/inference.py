import cv2
from detector import Detector
from detector_faster import DetectorFaster
import argparse
import time


def get_args():
    arg = argparse.ArgumentParser()
    arg.add_argument('-i', '--image_path', help='link to image', default='./corner_test.png')
    arg.add_argument('-o', '--option', help='detection or detection_faster', default='detection')
    return arg.parse_args()


arguments = get_args()
image_path = arguments.image_path
option = arguments.option

if option not in ['detection', 'detection_faster']:
    raise ValueError('Value option must be "detection" or "detection_faster"')

if option == "detection":
    detector = Detector(path_config='./ssd_mobilenet_v2/pipeline.config', path_ckpt='./ssd_mobilenet_v2/ckpt/ckpt-48',
                        path_to_labels='./scripts/label_map.pbtxt')
else:
    detector = DetectorFaster(path_to_model='./ssd_mobilenet_v2/exported_model/saved_model', path_to_labels='./scripts/label_map.pbtxt')

image = cv2.imread(image_path)
start = time.time()
image = detector.predict(image)
end = time.time()
print('Elapsed time: ', end - start)
cv2.imshow('results', image)
cv2.waitKey(0)
