corner_detection = {
    'path_to_model': './src/detector/config_corner_detection/model.tflite',
    'path_to_labels': './src/detector/config_corner_detection/label_map.pbtxt',
    'nms_ths': 0.2,
    'score_ths': 0.3
}

text_detection = {
    'path_to_model': './src/detector/config_text_detection/model.tflite',
    'path_to_labels': './src/detector/config_text_detection/label_map.pbtxt',
    'nms_ths': 0.2,
    'score_ths': 0.2

}

text_recognition = {
    'base_config': './src/vietocr/config_text_recognition/base.yml',
    'vgg_config': './src/vietocr/config_text_recognition/vgg-transformer.yml',
    'model_weight': './src/vietocr/config_text_recognition/transformerocr.pth'
}
