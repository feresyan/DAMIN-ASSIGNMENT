import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataTrain = pd.read_csv("TRAIN.csv")
dataTrain.shape
dataTest = pd.read_csv("TEST.csv")
dataTest.shape

X_train = dataTrain.drop('CLASS', axis=1)
y_train = dataTrain['CLASS']
X_test = dataTest.drop('CLASS', axis=1)
y_test = dataTest['CLASS']

print("DATA TRAIN")
print(X_train,"\n\n", y_train,"\n\n")

print("DATA TEST")
print(X_test,"\n\n", y_test,"\n\n")

from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)

print("PREDICT")
print(y_pred, "\n\n")

from sklearn.metrics import classification_report, confusion_matrix
temp = confusion_matrix(y_test,y_pred)
print(temp)

accuracy = (temp[0][0] + temp[1][1]) / 19
print(classification_report(y_test,y_pred))
print("accuracy : ", accuracy)