## This is a Random Forest Model on top of Kaggel Dataset "Telco Customer Churn". Accuracy Achieved - 80%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
df = pd.read_csv('attrition.csv')
#df['TotalCharges'].replace(r'\s+', np.nan, regex=True,inplace=True)
X = df.iloc[:,1:20]
y = df.iloc[:,-1:]
# Encoder method is used to convert the Numerical Values
Labelencoder_X = LabelEncoder()
encodelist = [0,2,3,5,6,7,8,9,10,11,12,13,14,15,16]
for i in encodelist:
    X.iloc[:,i] = Labelencoder_X.fit_transform(X.iloc[:,i])

Labelencoder_y = LabelEncoder()
#Labelencoder_y.fit_transform(y)
y['Churn']=Labelencoder_y.fit_transform(y['Churn'])

#Onehotencoder is not needed for Random Forest 
# Imputer is used to fill the missing values
#from sklearn.preprocessing import Imputer
#imputer = Imputer(missing_values= "nan",strategy="mean",axis=0)
#imputer = imputer.fit(X.iloc[:,-1:])
#X.iloc[:,-1:] = imputer.transform(X.iloc[:,-1:])

#Traning and Test data split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

#Random Forest regression
from sklearn.ensemble import RandomForestClassifier
randomforest = RandomForestClassifier(n_estimators=40,criterion='entropy',random_state=30)
randomforest.fit(X_train,y_train)

y_pred = randomforest.predict(X_test)

# print result
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_pred))
