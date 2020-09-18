import numpy as np
import tensorflow as tf
import cv2
from object_detection.utils import label_map_util


class DetectorFaster(object):
    def __init__(self, path_to_model, path_to_labels):
        self.path_to_model = path_to_model
        self.path_to_labels = path_to_labels
        self.category_index = label_map_util.create_category_index_from_labelmap(path_to_labels, use_display_name=True)

        # load model
        self.interpreter = self.load_model()
        # Get input and output tensors.
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.detection_scores = None
        self.detection_boxes = None
        self.detection_classes = None

    def load_model(self):
        # Load the TFLite model and allocate tensors.
        interpreter = tf.lite.Interpreter(model_path=self.path_to_model)
        interpreter.allocate_tensors()

        return interpreter

    def predict(self, img):
        height = self.input_details[0]['shape'][1]
        width = self.input_details[0]['shape'][2]
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        img = np.expand_dims(img, axis=0)
        # input_data = np.array(img, dtype=np.float32)

        input_mean = 127.5
        input_std = 127.5
        input_data = (np.float32(img) - input_mean) / input_std
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)

        self.interpreter.invoke()

        # Retrieve detection results
        self.detection_boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]  # Bounding box coordinates of detected objects
        self.detection_classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]  # Class index of detected objects
        self.detection_scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]  # Confidence of detected objects
        print("boxes: ", self.detection_boxes)
        print('scores: ', self.detection_scores)
        print('classes: ', self.detection_classes)
        # draw bounding boxes and labels
        image = self.draw(img[0])
        return image

    def draw(self, image):
        height, width, _ = image.shape
        for i, score in enumerate(self.detection_scores):
            if score < 0.2:
                continue

            self.detection_classes[i] += 1

            label = str(self.category_index[self.detection_classes[i]]['name'])
            real_ymin = int(max(1, (self.detection_boxes[i][0] * height)))
            real_xmin = int(max(1, (self.detection_boxes[i][1] * width)))
            real_ymax = int(min(height, (self.detection_boxes[i][2] * height)))
            real_xmax = int(min(width, (self.detection_boxes[i][3] * width)))

            cv2.rectangle(image, (real_xmin, real_ymin), (real_xmax, real_ymax), (0, 255, 0), 2)
            cv2.putText(image, label, (real_xmin, real_ymin), cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 255), fontScale=0.5)

        return image
