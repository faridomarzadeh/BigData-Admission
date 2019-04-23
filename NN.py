import os
import sys
import copy
import time
import random
from statistics import mean
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor


df = pd.read_csv('../data/Admission_Predict_Ver1.1.csv')

#Normalize dataframe

minmax_scaler=preprocessing.MinMaxScaler()
minmax_scaler_fit=minmax_scaler.fit(df[['GRE Score','TOEFL Score','University Rating','CGPA','SOP','LOR ','Research']])
NormalizedGREScoreAndTOEFLScore=minmax_scaler_fit.transform(df[['GRE Score','TOEFL Score','University Rating','CGPA','SOP','LOR ','Research']])
data=pd.DataFrame(NormalizedGREScoreAndTOEFLScore,columns=['GRE Score','TOEFL Score','University Rating','CGPA','SOP','LOR ','Research'])
df['GRE Score']=data['GRE Score']
df['TOEFL Score']=data['TOEFL Score']

df['CGPA']=data['CGPA']
df['University Rating']=data['University Rating']
df['SOP']=data['SOP']
df['LOR ']=data['LOR ']
df['Research']=data['Research']


predictorColumns=list(df.columns)
predictorColumns.remove('Serial No.')
predictorColumns.remove('Chance of Admit')

X=df[predictorColumns].values
y=df['Chance of Admit'].values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=10)




#Nerual Network
layers=np.arange(2,60,2)

res=[]
for num in layers :
    model=MLPRegressor(hidden_layer_sizes=num,random_state=10)
    model.fit(X_train,y_train)
    PreAdmit=model.predict(X_test)
    accuracyNN=r2_score(y_test,PreAdmit)
    res.append(accuracyNN)
plt.xlabel("number of neurons")
plt.ylabel("r2_score")
plt.plot(layers,res,'ro')
plt.axes([0,60,0,1])
plt.show()