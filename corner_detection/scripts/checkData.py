import os
from pathlib import Path
import cv2
import shutil


path_to_data = Path('./4CornersDatasets/')
i = 0
for path in path_to_data.glob('*.jpg'):
    img_name = path.name
    img = cv2.imread(str(path))

    if os.path.isfile(str(path)[:-4] + '.xml'):
        cv2.imwrite('./Dataset/img_' + str(i) + '.jpg', img)
        shutil.move(str(path)[:-4] + '.xml', './Dataset/img_' + str(i) + '.xml')
    else:
        cv2.imwrite('./NoLabel/img_' + str(i) + '.jpg', img)
    i += 1

print('Total: ', i)