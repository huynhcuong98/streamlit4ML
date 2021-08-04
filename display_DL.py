import streamlit as st 
import numpy as np 
import cv2

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from streamlit.elements import button, image
# from streamlit.proto.Image_pb2 import Image

from PIL import Image, ImageEnhance

from DL_project.Haar_cascade import Deploy_HaarCascade

def enhance_img(img):
    enhance_type = st.sidebar.radio("Enhance Type",["Original","Gray-Scale","Blurring"])

    if enhance_type == 'Gray-Scale':
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    elif enhance_type == 'Blurring':
        blur_rate = st.sidebar.slider("Brightness",0.5,3.5)
        img = cv2.GaussianBlur(img,(11,11),blur_rate)


    return img

def Run_HaarCascade():
    file = st.file_uploader("Upload an image", type=["png", "jpg"])
    if file is None:
        st.text("Please upload your image!!!")
    else:
        task = ["Faces","Smiles","Eye","body"]
        feature_choice = st.sidebar.selectbox("Find Features",task)
        img_raw = Image.open(file)
        img_raw = np.array(img_raw.convert('RGB'))
        img_raw = enhance_img(img_raw)
        st.image(img_raw, caption='Original Image')
        if(st.button("Process")):
            result_img, numb_face = Deploy_HaarCascade(img_raw, feature_choice)
            st.image(result_img, caption='Face detection')
            st.success("Found {} object".format(len(numb_face)))

def display_DL():
    project = st.sidebar.selectbox("Project", ("Haar Cascade", "Classification"))
    st.write("""
	## First implement with haar cascade
	""")
    if project == 'Haar Cascade':
        Run_HaarCascade()

    
