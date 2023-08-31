#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 11:50:11 2023

@author: Yannick
"""

# Importing the libraries
import numpy as np
import pandas as pd
import tensorflow as tf
tf.__version__

# Part 1 - Data Preprocessing

# Importing the dataset
dataset = pd.read_csv('/home/harry/Documents/Cours_deepLearning/ANN/Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values
print(X)
print(y)

# Encoding categorical data
# Label Encoding the "Gender" column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])
print(X)
# One Hot Encoding the "Geography" column
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Importation des modules Keras
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Validation croisée
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():
    # Initialiser l'ANN

    classifier = Sequential()

    #Ajouter des couches cachées
    classifier.add(Dense(units=6, activation='relu',
                         kernel_initializer='uniform', input_dim=12))
    
    #regularisation droupout (eviter overfitting)
    classifier.add(Dropout(rate=0.1))

    #Ajouter une deuxième couches cachées
    classifier.add(Dense(units=6, activation='relu',
                         kernel_initializer='uniform'))
    #regularisation droupout (eviter overfitting)
    classifier.add(Dropout(rate=0.1))

    #Ajouter la couche de sortie
    classifier.add(Dense(units=1, activation='sigmoid', 
                         kernel_initializer='uniform'))

    #Compiler l'ANN
    classifier.compile(optimizer='adam', loss='binary_crossentropy',
                       metrics=["accuracy"])
    
    return classifier

classifier1 = KerasClassifier(build_fn=build_classifier(), batch_size=10, epochs=100)

precisions = cross_val_score(estimator=classifier1, X=X_train, y=y_train, cv=10)

moyenne = precisions.mean()
ecart_type = precisions.std()


    




