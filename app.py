import streamlit as st
import numpy as np
from PIL import Image
from merged_model import CompletedModel
import time

st.set_page_config(layout="wide")
st.markdown('<center><h1 style="color: blue;">SCAN ID CARD</h1></center>',
                          unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Files", type=['png','jpeg', 'jpg'])
if uploaded_file is not None:
  title_container = st.beta_container()
  col1, col2, col3, col4 = st.beta_columns((0.6, 0.8, 0.8, 1))

  with title_container:       
    with col1:
      image = Image.open(uploaded_file) 
      img_np = np.array(image)

      st.image(image, caption='ID-Card for detection', width=200)
      # st.write("")
      # st.write("Running...")
      model = CompletedModel()
      start = time.time()
      img_crop, img_text, result = model.predict(img_np)
      end = time.time()
      total_time = end - start
    
    with col2:
      st.image(Image.fromarray(img_crop), caption='Cropped image', width=250)

    with col3:
      st.image(Image.fromarray(img_text), caption='Text recognition', width=250)
      
    with col4:
      st.write(result)
      st.write('Process time: ' + str(round(total_time, 2)) + 'seconds')
      st.markdown('<h3 style="color: green;">Scan ID-Card done !!!</h3>',
                          unsafe_allow_html=True)