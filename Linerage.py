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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('../data/Admission_Predict_Ver1.1.csv')

minmax_scaler=preprocessing.MinMaxScaler()
minmax_scaler_fit=minmax_scaler.fit(df[['GRE Score','TOEFL Score']])
NormalizedGREScoreAndTOEFLScore=minmax_scaler_fit.transform(df[['GRE Score','TOEFL Score']])
data=pd.DataFrame(NormalizedGREScoreAndTOEFLScore,columns=['GRE Score','TOEFL Score'])
df['GRE Score']=data['GRE Score']
df['TOEFL Score']=data['TOEFL Score']

predictorColumns=list(df.columns)
predictorColumns.remove('Serial No.')
predictorColumns.remove('Chance of Admit')
X=df[predictorColumns].values
y=df['Chance of Admit'].values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=10)

model=LinearRegression()
model.fit(X_train,y_train)
PreAdmit=model.predict(X_test)
AdmitData=pd.DataFrame(X_test,columns=predictorColumns)
AdmitData['ChanceOfAdmit']=y_test
AdmitData['PredictedChancesOfAdmit']=PreAdmit

#rmse=np.sqrt(mean_squared_error(y_test,PreAdmit))
accuracy=r2_score(y_test,PreAdmit)
print(accuracy)
#print(rmse)
