from pathlib import Path
import xml.etree.ElementTree as ET
import os


dir_xml = Path('./Dataset/')
i = 0
for xml_path in dir_xml.glob("*.xml"):
    mytree = ET.parse(xml_path)
    name = xml_path.name
    myroot = mytree.getroot()

    myroot.find('filename').text = name[:-4] + '.jpg'
    mytree.write(xml_path)
    i += 1

print('total: ', i)
