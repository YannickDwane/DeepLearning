#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 15:28:30 2023

@author: Yannick
"""

# preparation des données

# imporation des packages

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("/home/malina/Documents/Cours_deepLearning/data_preprocessing/Data.csv")

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# ajout des valeurs manquante

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

imputer.fit(X[:, 1:3])

X[:, 1:3] = imputer.transform(X[:, 1:3])
print(X)

#Transformation des variable de catégorie
#Variable indépendant
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer


ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], 
                         remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)

#variable dépendant
le = LabelEncoder()

y = le.fit_transform(y)
print(y)

# separer les donées en jeu d'entrainement et en jeu de test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)
print(X_train)
print(y_train)
print(X_test)
print(y_test)

#changement d'echelle
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train= sc.fit_transform(X_train)
X_test = sc.transform(X_test)









