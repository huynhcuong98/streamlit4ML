import streamlit as st 
import numpy as np 
# import pandas as pd 

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from display_ML import display_ML
from display_DL import display_DL

def Process_ML_page():
	st.title('Machine Learning')
	method = st.sidebar.selectbox(
	'Choose method',
	('Your Dataset', 'Example Dataset'))

	if method == 'Example Dataset':
		datatype = st.sidebar.selectbox(
			'Select type of data',
			('Number', 'Image'))
		if datatype == 'Number':
			display_ML()
		else:
			dataset_name = st.sidebar.selectbox(
				'Select Dataset',
				('MNIST', 'Cifar', )
			)

def Process_DL_page():
	st.title('Deep Learning')
	display_DL()





type = st.sidebar.selectbox(
	'Choose Type',
	('Machine Learning', 'Deep learning'))

if type == 'Machine Learning':
	Process_ML_page()
else:
	Process_DL_page()






		
