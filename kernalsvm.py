# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 13:03:35 2019

@author: user
"""
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 15:06:54 2019

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 15:05:56 2019

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 14:54:04 2019

@author: user
"""

import pandas as pd
import numpy as np
import matplotlib as ml
import seaborn as sns

dataset = pd.read_csv('Social_Network_Ads.csv')

print(dataset)

X = dataset.iloc[:, 2:4]
y = dataset.iloc[:, 4:5]

print(X)
print(y)
print(type(X))
print(type(y))


#Assigning data for test and training,if test is 25% then by defalut traing is 75
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=.25,random_state=20)
# if we are using StandardScaler then use dataframe.
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train= scaler.transform(X_train)
scaler.fit(X_test)
X_test = scaler.transform(X_test)
#Apllying model to dataset,here random_state value can be any value.

from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

prediction=classifier.predict(X_test)
#sns.distplot(y_test-prediction)
print(prediction)

#print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, prediction)


from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
print ('Accuracy Score :',accuracy_score(y_test, prediction) )
