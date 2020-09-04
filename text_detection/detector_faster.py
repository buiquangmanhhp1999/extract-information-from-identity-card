import tensorflow as tf
from object_detection.utils import label_map_util
import cv2
import numpy as np


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
        detection_model = tf.saved_model.load(self.path_to_model)
        detection_model = detection_model.signatures['serving_default']
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
        image = self.draw(image)
        return image

    def draw(self, image):
        height, width, _ = image.shape
        for i, score in enumerate(self.detection_scores):
            if score < 0.5:
                continue

            # if background, ignore
            if self.detection_classes[i] == 0:
                continue

            label = str(self.category_index[self.detection_classes[i]]['name'])
            ymin, xmin, ymax, xmax = self.detection_boxes[i]
            real_xmin, real_ymin, real_xmax, real_ymax = int(xmin * width), int(ymin * height), int(xmax * width), int(ymax * height)
            cv2.rectangle(image, (real_xmin, real_ymin), (real_xmax, real_ymax), (0, 255, 0), 2)
            cv2.putText(image, label, (real_xmin, real_ymin), cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 255), fontScale=0.5)

        return image
