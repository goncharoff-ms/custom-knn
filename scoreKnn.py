# coding=utf-8
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor

from knnLib.KNNClasses import KNNClassifier
from knnLib.KNNClasses import KNNRegressor


data = pd.read_csv("car.csv")

feature_names = ['buing', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
X = data[feature_names].astype(int).values
y = data['class'].astype(int).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

print("My Classifier")
classifier = KNNClassifier(20, X_train, y_train)
prediction = classifier.predict(X_test)
print(accuracy_score(y_test, prediction))
print("sklearn Classifier")
classifier = KNeighborsClassifier(n_neighbors=20)
classifier.fit(X_train, y_train)
prediction = classifier.predict(X_test)
print(accuracy_score(y_test, prediction))

dataRegg = pd.read_csv("Container_Crane_Controller_Data_Set.csv")

data_train = dataRegg.iloc[:120]
data_test = dataRegg.iloc[120:]

X_train = data_train[['m', 's', 'p', 'v']].astype(float).values
y_train = data_train[['class']].astype(float).values

X_test = data_test[['m', 's', 'p', 'v']].astype(float).values
y_test = data_test[['class']].astype(float).values

print("\n")
print("My KNNREgresoor")
regressor = KNNRegressor(15, X_train, y_train)
prediction = regressor.predict(X_test)
print(mean_absolute_error(y_test, prediction))

print("sklearn KNNRegressor")
regressor = KNeighborsRegressor(n_neighbors=4)
regressor.fit(X_train, y_train)
prediction = regressor.predict(X_test)
print(mean_absolute_error(y_test, prediction))
