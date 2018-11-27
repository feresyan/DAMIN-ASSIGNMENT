import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

bankdata = pd.read_csv("TRAIN Normalisasi.csv")
bankdata.shape

X = bankdata.drop('CLASS', axis=1)
y = bankdata['CLASS']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.23,shuffle=False)

print("DATA TRAIN")
print(X_train,"\n\n", y_train,"\n\n")

print("DATA TEST")
print(X_test,"\n\n", y_test,"\n\n")

from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
temp = confusion_matrix(y_test,y_pred)
print(temp)

accuracy = (temp[0][0] + temp[1][1]) / 10
print(classification_report(y_test,y_pred))
print("accuracy : ", accuracy)