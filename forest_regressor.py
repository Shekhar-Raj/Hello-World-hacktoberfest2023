// How to use random Forest Regressor in Machine learning

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_boston        // import dataset

df=load_boston()
df

df['feature_names']                            // getting features  name

data=pd.DataFrame(df.data,columns=df.feature_names)
data.head()

data['MEDV']=df.target
data

data.isnull().sum()                           // checking null values

data.describe()                               // describing the dataset like median mode, maximum, minimum

 // Splitting the dataset

x=data.drop(['MEDV'],axis=1)
x

y=data['MEDV']
y

// Using Standard Scaler

from sklearn.preprocessing import StandardScaler

sc_x=StandardScaler()
x=sc_x.fit_transform(x)

//Training and Testing the dataset

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=42)

ytrain.head()

ytest.head()

//Importing RandomForestRegressor

from sklearn.ensemble import RandomForestRegressor
r_f=RandomForestRegressor(n_estimators=500)
r_f.fit(xtrain,ytrain)

y_pred=r_f.predict(xtest)

//Here the error becomes half the error we got in decision Tree Regression.

from sklearn.metrics import mean_squared_error
mean_squared_error(ytest,y_pred)
