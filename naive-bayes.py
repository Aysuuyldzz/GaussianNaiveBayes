# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd 
from sklearn import preprocessing

data=pd.read_excel('Immunotherapy1.xlsx')


data=data.apply(preprocessing.LabelEncoder().fit_transform)

x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.15,random_state=0)


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()

x_train=scaler.fit_transform(x_train)
x_test=scaler.fit_transform(x_test)
y_train=scaler.fit_transform(y_train.reshape(-1,1))
y_test=scaler.fit_transform(y_test.reshape(-1,1))

 
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)
y_pred

from sklearn.metrics import confusion_matrix,classification_report
hm=confusion_matrix(y_test, y_pred)
rapor=classification_report(y_test, y_pred)

from sklearn.metrics import roc_curve,auc

ypo,dpo,esikDeger=roc_curve(y_test,y_pred)
aucDegeri=auc(ypo,dpo)
plt.figure()
plt.plot(ypo,dpo,label='AUC %0.2f'%aucDegeri)
plt.plot([0,1],[0,1],'k--')
plt.legend(loc="best")
plt.show()