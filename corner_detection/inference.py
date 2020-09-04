import cv2
from detector import Detector
import time
import argparse
from detect_utils import align_image


def get_args():
    arg = argparse.ArgumentParser()
    arg.add_argument('-i', '--image_path', help='link to image', default='./test2.jpg')
    return arg.parse_args()


arguments = get_args()
image_path = arguments.image_path

detector = Detector(path_config='./ssd_mobilenet_v2/pipeline.config', path_ckpt='./ssd_mobilenet_v2/ckpt/ckpt-49',
                    path_to_labels='./scripts/label_map.pbtxt')


image = cv2.imread(image_path)
start = time.time()
image, original_image, coordinate_dict = detector.predict(image)
end = time.time()
print("Predicted time: ", end - start)
img = align_image(original_image, coordinate_dict)
cv2.imwrite('corner_test.png', img)
cv2.imshow('test', img)
cv2.waitKey(0)