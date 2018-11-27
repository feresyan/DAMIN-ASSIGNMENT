import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

bankdata = pd.read_csv("TRAIN Preprocessing.csv")
bankdata.shape

X = bankdata[['ATR3','ATR1']]
y = bankdata['CLASS']

print(X.max())
print(X.min())

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

w = svclassifier.coef_[0]

a = -w[0] / w[1]

xx = np.linspace(39.38125,3321.2450) # angka terkecil dan terbesar dalam data
yy = a * xx - svclassifier.intercept_[0] / w[1]

h0 = plt.plot(xx,yy,'blue')

klasifikasi_0_X = []
klasifikasi_1_X = []
klasifikasi_0_y = []
klasifikasi_1_y = []

klasifikasi_0_X.append(bankdata[33:].loc[bankdata['CLASS'] == 'NEGATIVE']['ATR3'])
klasifikasi_1_X.append(bankdata[33:].loc[bankdata['CLASS'] == 'POSITIVE']['ATR3'])
klasifikasi_0_y.append(bankdata[33:].loc[bankdata['CLASS'] == 'NEGATIVE']['ATR1'])
klasifikasi_1_y.append(bankdata[33:].loc[bankdata['CLASS'] == 'POSITIVE']['ATR1'])


negative = plt.scatter(klasifikasi_0_X,klasifikasi_0_y,color='red')
positive = plt.scatter(klasifikasi_1_X,klasifikasi_1_y,color='blue')

plt.legend((negative,positive),
           ("Kelas Negatif", "Kelas Positif"),
           loc='upper center',
           ncol=6,
           fontsize=8,
           bbox_to_anchor=(0.5, -0.05))

plt.show()