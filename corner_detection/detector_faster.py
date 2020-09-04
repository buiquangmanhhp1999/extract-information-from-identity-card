import tensorflow as tf
from object_detection.utils import label_map_util
import cv2
import numpy as np
import time
from tensorflow.keras.models import save_model


class DetectorFaster(object):
    def __init__(self, path_to_model, path_to_labels):
        self.path_to_model = path_to_model
        self.path_to_labels = path_to_labels

        self.category_index = label_map_util.create_category_index_from_labelmap(path_to_labels, use_display_name=True)
        self.detection_model = self.load_model()
        self.detection_scores = None
        self.detection_boxes = None
        self.detection_classes = None

    def load_model(self):
        start_load_model = time.time()
        detection_model = tf.saved_model.load(self.path_to_model)
        save_model(detection_model, self.path_to_model + r"\new_model.h5", save_format='h5')
        end_load_model = time.time()
        print("Loaded model time: ", end_load_model - start_load_model)
        return detection_model

    def predict(self, image):
        image = np.asarray(image)
        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        input_tensor = tf.convert_to_tensor(image)
        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        input_tensor = input_tensor[tf.newaxis, ...]

        # Run inference
        detections = self.detection_model(input_tensor)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        self.detection_scores = detections['detection_scores']
        self.detection_classes = detections['detection_classes']
        self.detection_boxes = detections['detection_boxes']

        # draw bounding boxes and labels
        image, coordinate_dict = self.draw(image)
        return image, coordinate_dict

    def draw(self, img):
        coordinate_dict = dict()
        height, width, _ = img.shape
        li = []

        for i, score in enumerate(self.detection_scores):
            if score < 0.3:
                continue

            # if background, ignore
            if self.detection_classes[i] == 0:
                continue

            label = str(self.category_index[self.detection_classes[i]]['name'])
            ymin, xmin, ymax, xmax = self.detection_boxes[i]
            real_xmin, real_ymin, real_xmax, real_ymax = int(xmin * width), int(ymin * height), int(xmax * width), int(
                ymax * height)

            curr = real_xmax * real_ymax - real_ymin * real_xmin
            status = check_overlap(curr, li)
            if status == 1:
                continue

            li.append(real_xmax * real_ymax - real_ymin * real_xmin)
            # check overlap bounding boxes

            cv2.rectangle(img, (real_xmin, real_ymin), (real_xmax, real_ymax), (0, 255, 0), 2)
            cv2.putText(img, label, (real_xmin, real_ymin), cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 255), fontScale=0.5)
            coordinate_dict[label] = (real_xmin, real_ymin, real_xmax, real_ymax)

        return img, coordinate_dict


def check_overlap(curr, li):
    for va in li:
        # overlap
        if abs(va - curr) < 1000:
            return 1
    return 0


model = DetectorFaster(path_to_model='./ssd_mobilenet_v2/saved_model/', path_to_labels='scripts/label_map.pbtxt')
image = cv2.imread('./test2.jpg')
start = time.time()
image, coordinate_dict = model.predict(image)
# print(coordinate_dict)
end = time.time()
print("Predicted time: ", end - start)
cv2.imshow('test', image)
cv2.waitKey(0)
