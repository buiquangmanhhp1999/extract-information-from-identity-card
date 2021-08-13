# Extract-information-from-identity-card
Tutorial: [Recognize Identity Card Tutorial](https://viblo.asia/p/trich-xuat-thong-tin-tu-chung-minh-thu-bJzKmaRwK9N)

## 1. Pipeline

### 1.1. Corner Detection and Alignment
<p align="center">
  <img width="800" height="350" src="https://user-images.githubusercontent.com/48142689/92223664-fd60b780-eeca-11ea-8e7e-76f93f4ed888.png">
</p>

### 1.2. Text Detection
<p align="center">
  <img width="800" height="350" src="https://user-images.githubusercontent.com/48142689/92224160-a0193600-eecb-11ea-9243-82d02d86812a.png">
</p>

### 1.3. Recognize image text and final results
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

## 2. How to run project 

### 2.1. Install dependecies
```
pip install -r requirement.txt
```
### 2.2. Download weight from gg drive

Download weight from [here](https://drive.google.com/file/d/1pXftFiTGzcXNqsy6jKxQF2WyiOoBmDKU/view?usp=sharing) , then put into src/vietocr/config_text_recognition folder.

### 2.3. Start server
```
python main.py
```





