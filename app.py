import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.title('Exploring different datasets')
st.header('Let us see which model works best for one dataset')

#selection for datasets
dataset_name = st.sidebar.selectbox('Select Dataset' , ('Iris', 'Breast Cancer', 'Wine'))

#selection for classifier
classifier_name = st.sidebar.selectbox('Select Classifier', ('KNN', 'SVM', 'Random Forest'))

def get_dataset(dataset_name):
    data = None
    if dataset_name == 'Iris':
        data = datasets.load_iris()
    elif dataset_name == 'Wine':
        data = datasets.load_wine()
    else:
        data = datasets.load_breast_cancer()
    
    X = data.data
    y = data.target

    return X,y

X, y = get_dataset(dataset_name)

st.write('Shape of dataset:', X.shape)
st.write('number of classes:', len(np.unique(y)))


def add_parameter_ui(classifier_name):
    params = dict()
    if classifier_name == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0) # degree of its correct classification
        params['C'] = C

    elif classifier_name == 'KNN':
        K = st.sidebar.slider('K', 1, 15) # k is number of nearest neighbor
        params['K'] = K

    else:
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        params['max_depth'] = max_depth # depth of every tree in random forest
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators # number of trees
    
    return params

params = add_parameter_ui(classifier_name)

def get_classifier(classifier_name, params):
    clf = None
    if classifier_name == 'SVM':
        clf = SVC(C=params['C'])
    elif classifier_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    else:
        clf = RandomForestClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'], random_state=1234)

    return clf

clf = get_classifier(classifier_name, params)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)

st.write(f'Dataset = {dataset_name}')
st.write(f'Classifier = {classifier_name}')
st.write(f'Accuracy = ', acc)

pca = PCA(2)
X_projected = pca.fit_transform(X)

X1 = X_projected[:, 0]
X2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(X1, X2, c=y, alpha=0.8, cmap='viridis')

plt.xlabel('Principal Complonent 1')
plt.ylabel('Principal Complonent 2')
plt.colorbar()

st.pyplot(fig)