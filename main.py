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
from display_ML import display1

st.title('Machine Learning class')

method = st.sidebar.selectbox(
	'Choose method',
	('Your Dataset', 'Ours Dataset'))


if method == 'Ours Dataset':
	datatype = st.sidebar.selectbox(
		'Select type of data',
		('Number', 'Image'))
	if datatype == 'Number':
		display1()
		
	else:
		dataset_name = st.sidebar.selectbox(
		    'Select Dataset',
		    ('MNIST', 'Cifar', )
		)
