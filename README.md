# Extract-information-from-identity-card

## 1. Install dependencies
```
cd complete
pip install -r requirement.txt
```

## 2. Corner Detection and Alignment
Run the above command
```
cd corner_detection
python inference.py --image_path=image_path
```
Results:
<p align="center">
  <img width="800" height="350" src="https://user-images.githubusercontent.com/48142689/92223664-fd60b780-eeca-11ea-8e7e-76f93f4ed888.png">
</p>

## 3. Text Detection
Run the above command
```
cd text_detection
python inference.py --image_path=image_path
```
Results:
<p align="center">
  <img width="800" height="350" src="https://user-images.githubusercontent.com/48142689/92224160-a0193600-eecb-11ea-9243-82d02d86812a.png">
</p>

## 3. End to end model (corner detection, text detection, ocr)
Run above command to run server
```
cd complete
python manage.py runserver 8080
```

**Final Result**
```
{
  "id": "38138424",
  "name": "LÊ KIỀU DIỄM",
  "birth": "1989",
  "add": "Tân Hưng Tây Phú Tân Cà Mau",
  "home": "Khóm 8 Phường 8 TP Cà Mau Cà Mau"
}
```
