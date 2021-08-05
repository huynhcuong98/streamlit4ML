import streamlit as st 
import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, plot_confusion_matrix, confusion_matrix
import webbrowser

def get_dataset(name):
	    data = None
	    if name == 'Iris':
	        data = datasets.load_iris()
	    elif name == 'Wine':
	        data = datasets.load_wine()
	    else:
	        data = datasets.load_breast_cancer()

	    X = data.data
	    y = data.target
	    #data: {data, target, target_names, DESCR, feature_names, filename}

	    return X, y, data

def create_dataframe(y,data):
	x = np.where(y==0, data.target_names[0], y)
	for i in range(1,len(np.unique(y))):
		print(i)
		x = np.where(x ==str(i), data.target_names[i] ,x)
	df = pd.DataFrame(data.data)
	df.columns = data.feature_names
	df.insert(len(data.feature_names), 'Target', x)
	return df 

def add_parameter_ui(clf_name):
	    params = dict()
	    if clf_name == 'SVM':
	        C = st.sidebar.slider('C', 0.01, 10.0)
	        kernel = st.sidebar.radio("Kernel",("rbf", "linear"), key='kernel')
	        gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key = 'gamma')
	        params['kernel'] = kernel
	        params['gamma'] = gamma
	        params['C'] = C
	    elif clf_name == 'KNN':
	        K = st.sidebar.slider('K', 1, 15)
	        params['K'] = K
	    else:
	        max_depth = st.sidebar.slider('max_depth', 2, 15)
	        params['max_depth'] = max_depth
	        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
	        params['n_estimators'] = n_estimators
	    return params

def get_classifier(clf_name, params):
	    clf = None
	    if clf_name == 'SVM':
	        clf = SVC(C=params['C'])
	    elif clf_name == 'KNN':
	        clf = KNeighborsClassifier(n_neighbors=params['K'])
	    else:
	        clf = clf = RandomForestClassifier(n_estimators=params['n_estimators'], 
	            max_depth=params['max_depth'], random_state=1234)
	    return clf

def plot_PCA(X,y):
	pca = PCA(2)
	X_projected = pca.fit_transform(X)

	x1 = X_projected[:, 0]
	x2 = X_projected[:, 1]


	fig = plt.figure()
	plt.scatter(x1, x2,
	        c=y, alpha=0.8,
	        cmap='viridis')

	plt.xlabel('Principal Component 1')
	plt.ylabel('Principal Component 2')
	plt.colorbar()

	st.pyplot(fig)
	#plt.show()

def get_url(name):
	if name == 'SVM':
		url = 'https://machinelearningcoban.com/2017/04/09/smv/'
	elif name == 'KNN':
		url = 'https://machinelearningcoban.com/2017/01/08/knn/'
	else:
		url = 'https://machinelearningcoban.com/tabml_book/ch_model/random_forest.html'
	return url

def plot_cfs_matrix(clf, X_test, y_test, class_names):

	st.subheader("Confusion Matrix") 
	plot_confusion_matrix(clf, X_test, y_test, display_labels=class_names) #display_labels
	st.set_option('deprecation.showPyplotGlobalUse', False)
	st.pyplot()


def display_ML():
	dataset_name = st.sidebar.selectbox(
		    'Select Dataset',
		    ('Iris', 'Breast Cancer', 'Wine')
		)
	
	st.write(f"## {dataset_name} Dataset")
	st.subheader("Data infomation")
	classifier_name = st.sidebar.selectbox(
	    'Select classifier',
	    ('KNN', 'SVM', 'Random Forest')
	)
	X, y, breaf= get_dataset(dataset_name)
	st.write("Data:")
	df =create_dataframe(y, breaf)
	st.dataframe(df)
	fd = np.array(breaf.feature_names)
	fd= fd.reshape(1,-1)
	st.write('number of features:', len(breaf.feature_names))
	st.write('features:', fd)
	st.write('number of classes:', len(np.unique(y)))
	st.write('classes:', np.array(breaf.target_names).reshape(1,-1))

#### PLOT DATASET ####
	# Project the data onto the 2 primary principal components
	st.write("## ANALYS DATA BY PCA ")
	plot_PCA(X,y)
	

	
	#### CLASSIFICATION ####
	params = add_parameter_ui(classifier_name)
	clf = get_classifier(classifier_name, params)

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)

	acc = accuracy_score(y_test, y_pred)
	st.write("## EVALUATION ")
	# st.write(f'Accuracy =', acc)
	st.subheader(f"Accuracy = {acc}") 
	plot_cfs_matrix(clf, X_test, y_pred, breaf.target_names)
	
	###		REF 		####
	st.write("## REFERENCE ")
	st.write(f'Link {classifier_name}:')
	if st.button('Open lesson'):
		url = get_url(classifier_name)
		webbrowser.open_new_tab(url)
	st.write('Source code:')
	if st.button('Open source'):
		url = 'https://github.com/huynhcuong98/CP_AIcode'
		webbrowser.open_new_tab(url)


	
