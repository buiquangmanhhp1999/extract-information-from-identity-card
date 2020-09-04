from pathlib import Path
import xml.etree.ElementTree as ET
import os


dir_xml = Path('./TrainValDataset/test/')
i = 0
for xml_path in dir_xml.glob("*.xml"):
    mytree = ET.parse(xml_path)
    name, extension = os.path.splitext(xml_path.name)
    myroot = mytree.getroot()

    for object in mytree.iter('object'):
        if object.find('name').text == 'image':
            i += 1
            object.clear()
    mytree.write(xml_path)

print(i)