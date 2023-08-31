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

# Initialiser l'ANN

classifier = Sequential()

# #Ajouter des couches cachées
classifier.add(Dense(units=6, activation='relu',
                      kernel_initializer='uniform'))

# #Ajouter une deuxième couches cachées
classifier.add(Dense(units=6, activation='relu',
                      kernel_initializer='uniform'))

# #Ajouter la couche de sortie
classifier.add(Dense(units=1, activation='sigmoid', 
                      kernel_initializer='uniform'))

# #Compiler l'ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy',
                    metrics=["accuracy"])

# #Entrainer l'ANN
classifier.fit(X_train,y_train, batch_size=10, epochs=100)

#Prédiction sur le jeu de test
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

#Générer la matrice de confusion
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# #Prédiction pour un client
""" Pays : France
Score de credits : 600
Genre : Male
Age : 40
Dureee depuis entrée dans la banque : 3 ans
Balance : 60 000
Nombre de produits: 2
Carte de credit ? 2
Melbre actif ? 2
Salaire estime : 50 000 €

"""

new_prediction = classifier.predict(sc.transform(np.array([[1, 0.0, 0.0, 600, 0.0, 40, 3, 60000, 2, 1, 1, 50000]])))
new_predictiton = ( new_prediction > 0.5 )




    




