from detector import Detector
from recognition import TextRecognition
from utils.image_utils import align_image, sort_text
import cv2
import time
import numpy as np
from copy import deepcopy as dc


class CompletedModel(object):
    def __init__(self):
        self.corner_detection_model = Detector(path_to_model='[root]/extract-information-from-identity-card/complete/config_corner_detection/model.tflite',
                                               path_to_labels='[root]/extract-information-from-identity-card/complete/config_corner_detection/label_map.pbtxt',
                                               nms_threshold=0.2, score_threshold=0.3)
        self.text_detection_model = Detector(path_to_model='[root]/extract-information-from-identity-card/complete/config_text_detection/model.tflite',
                                             path_to_labels='[root]/extract-information-from-identity-card/complete/config_text_detection/label_map.pbtxt',
                                             nms_threshold=0.2, score_threshold=0.2)
        self.text_recognition_model = TextRecognition(path_to_checkpoint='[root]/transformerocr.pth') #download link: https://drive.google.com/file/d/1B7b52G0hL6SKpsxP70xfyKhaRPaoWTuJ/view?usp=sharing

        # init boxes
        self.id_boxes = None
        self.name_boxes = None
        self.birth_boxes = None
        self.add_boxes = None
        self.home_boxes = None

    def detect_corner(self, image):
        detection_boxes, detection_classes, category_index = self.corner_detection_model.predict(image)

        coordinate_dict = dict()
        height, width, _ = image.shape

        for i in range(len(detection_classes)):
            label = str(category_index[detection_classes[i]]['name'])
            real_ymin = int(max(1, detection_boxes[i][0]))
            real_xmin = int(max(1, detection_boxes[i][1]))
            real_ymax = int(min(height, detection_boxes[i][2]))
            real_xmax = int(min(width, detection_boxes[i][3]))
            coordinate_dict[label] = (real_xmin, real_ymin, real_xmax, real_ymax)

        # align image
        cropped_img = align_image(image, coordinate_dict)
        # cv2.imwrite('./test.png', cropped_img)

        return cropped_img

    def detect_text(self, image):
        # detect text boxes
        detection_boxes, detection_classes, category_index = self.text_detection_model.predict(image)

        # sort text boxes according to coordinate
        self.id_boxes, self.name_boxes, self.birth_boxes, self.home_boxes, self.add_boxes = sort_text(detection_boxes, detection_classes)

        return self.text_detection_model.draw(image)

    def text_recognition(self, image):
        field_dict = dict()

        # crop boxes according to coordinate
        def crop_and_recog(boxes):
            crop = []
            if len(boxes) == 1:
                ymin, xmin, ymax, xmax = boxes[0]
                crop.append(image[ymin:ymax, xmin:xmax])
            else:
                for box in boxes:
                    ymin, xmin, ymax, xmax = box
                    # cv2.imwrite('./crop/test_' + str(ymin) + '_' + str(ymax) + '.png', image[ymin:ymax, xmin:xmax])
                    crop.append(image[ymin:ymax, xmin:xmax])

            return crop

        list_ans = list(crop_and_recog(self.id_boxes))
        list_ans.extend(crop_and_recog(self.name_boxes))
        list_ans.extend(crop_and_recog(self.birth_boxes))
        list_ans.extend(crop_and_recog(self.add_boxes))
        list_ans.extend(crop_and_recog(self.home_boxes))

        start1 = time.time()
        result = self.text_recognition_model.predict_on_batch(np.array(list_ans))
        end1 = time.time()
        print("predicted time: ", end1 - start1)
        field_dict['id'] = result[0]
        field_dict['name'] = ' '.join(result[1:len(self.name_boxes) + 1])
        field_dict['birth'] = result[len(self.name_boxes) + 1]
        field_dict['home'] = ' '.join(result[len(self.name_boxes) + 2: -len(self.home_boxes)])
        field_dict['add'] = ' '.join(result[-len(self.home_boxes):])
        return field_dict

    def predict(self, image):
        cropped_image = self.detect_corner(image)
        cropped_image_cp = dc(cropped_image)
        text_image = self.detect_text(cropped_image)
        return cropped_image_cp, text_image, self.text_recognition(cropped_image)


if "__name__" == "__main__":
    model = CompletedModel()
    """
    image = request.args['image_url']
    """
    np_image = np.array(image)
    _, _, text = model.predict(np_image)