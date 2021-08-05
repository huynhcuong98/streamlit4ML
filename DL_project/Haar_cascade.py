import cv2
import numpy as np
import streamlit as st 

def Deploy_HaarCascade(img, feature):

    f_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    e_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    smile_cascade =  cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
    body_cascade =  cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")

    img_original = img.copy()
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    
    if feature == "Faces":
        obj = f_cascade.detectMultiScale(img, 1.3, 5)
        for (x,y,w,h) in obj:
            img_original = cv2.rectangle(img_original,(x,y),(x+w,y+h),(255,0,0),2)
    elif feature == "Smiles":
        obj = smile_cascade.detectMultiScale(img, 1.3, 5)
        for (x,y,w,h) in obj:
            img_original = cv2.rectangle(img_original,(x,y),(x+w,y+h),(255,0,0),2)
    elif feature == "Eyes":
        obj = e_cascade.detectMultiScale(img, 1.3, 5)
        for (x,y,w,h) in obj:
            img_original = cv2.rectangle(img_original,(x,y),(x+w,y+h),(255,0,0),2)
    elif feature == "body":
        obj = body_cascade.detectMultiScale(img, 1.3, 5)
        for (x,y,w,h) in obj:
            img_original = cv2.rectangle(img_original,(x,y),(x+w,y+h),(255,0,0),2)

        
        # roi_gray = gray[y:y+h, x:x+w]
        # roi_color = img[y:y+h, x:x+w]
        # eyes = e_cascade.detectMultiScale(roi_gray)
        # for (ex,ey,ew,eh) in eyes:
        #     cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    # cv2.imshow('img',image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows(
    return img_original, obj