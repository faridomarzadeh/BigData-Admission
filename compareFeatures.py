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
minmax_scaler_fit=minmax_scaler.fit(df[['GRE Score','TOEFL Score','CGPA']])
NormalizedGREScoreAndTOEFLScore=minmax_scaler_fit.transform(df[['GRE Score','TOEFL Score','CGPA']])
data=pd.DataFrame(NormalizedGREScoreAndTOEFLScore,columns=['GRE Score','TOEFL Score','CGPA'])
df['GRE Score']=data['GRE Score']
df['TOEFL Score']=data['TOEFL Score']
df['CGPA']=data['CGPA']

predictorColumns=list(df.columns)
predictorColumns.remove('Serial No.')
predictorColumns.remove('Chance of Admit')

X=df[predictorColumns].values
y=df['Chance of Admit'].values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=36)

clf=RandomForestRegressor(n_estimators=100,max_depth=45,criterion='mse',min_samples_leaf=12,random_state=36)
RF=clf.fit(X_train,y_train)
PreAdmit=RF.predict(X_test)
AdmitData=pd.DataFrame(X_test,columns=predictorColumns)
AdmitData['ChanceOfAdmit']=y_test
AdmitData['PredictedChancesOfAdmit']=PreAdmit
rmse=np.sqrt(mean_squared_error(y_test,PreAdmit))
accuracy=r2_score(y_test,PreAdmit)

model=LinearRegression()
model.fit(X_train,y_train)
PreAdmit=model.predict(X_test)
AdmitData=pd.DataFrame(X_test,columns=predictorColumns)
AdmitData['ChanceOfAdmit']=y_test
AdmitData['PredictedChancesOfAdmit']=PreAdmit
rmseLin=np.sqrt(mean_squared_error(y_test,PreAdmit))
accuracyLin=r2_score(y_test,PreAdmit)

#Deceision tree
model=DecisionTreeRegressor(random_state=36,max_depth=100,min_samples_leaf=10)
model.fit(X_train,y_train)
PreAdmit=model.predict(X_test)
AdmitData=pd.DataFrame(X_test,columns=predictorColumns)
AdmitData['ChanceOfAdmit']=y_test
AdmitData['PredictedChancesOfAdmit']=PreAdmit
rmseDT=np.sqrt(mean_squared_error(y_test,PreAdmit))
accuracyDT=r2_score(y_test,PreAdmit)



#KNN
model=KNeighborsRegressor(n_neighbors=9,metric='euclidean')
model.fit(X_train,y_train)
PreAdmit=model.predict(X_test)
AdmitData=pd.DataFrame(X_test,columns=predictorColumns)
AdmitData['ChanceOfAdmit']=y_test
AdmitData['PredictedChancesOfAdmit']=PreAdmit
rmseKNN=np.sqrt(mean_squared_error(y_test,PreAdmit))
accuracyKNN=r2_score(y_test,PreAdmit)



#Nerual Network
model=MLPRegressor(hidden_layer_sizes=50,random_state=10)
model.fit(X_train,y_train)
PreAdmit=model.predict(X_test)
AdmitData=pd.DataFrame(X_test,columns=predictorColumns)
AdmitData['ChanceOfAdmit']=y_test
AdmitData['PredictedChancesOfAdmit']=PreAdmit
rmseNN=np.sqrt(mean_squared_error(y_test,PreAdmit))
accuracyNN=r2_score(y_test,PreAdmit)

#3 features
predictorColumns=list(df.columns)
predictorColumns.remove('Serial No.')
predictorColumns.remove('Chance of Admit')
predictorColumns.remove('University Rating')
predictorColumns.remove('SOP')
predictorColumns.remove('LOR ')
predictorColumns.remove('Research')
X=df[predictorColumns].values
y=df['Chance of Admit'].values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=36)

clf=RandomForestRegressor(n_estimators=100,max_depth=45,criterion='mse',min_samples_leaf=12,random_state=36)
RF=clf.fit(X_train,y_train)
PreAdmit=RF.predict(X_test)
AdmitData=pd.DataFrame(X_test,columns=predictorColumns)
AdmitData['ChanceOfAdmit']=y_test
AdmitData['PredictedChancesOfAdmit']=PreAdmit
rmse2=np.sqrt(mean_squared_error(y_test,PreAdmit))
accuracy2=r2_score(y_test,PreAdmit)


model=LinearRegression()
model.fit(X_train,y_train)
PreAdmit=model.predict(X_test)
AdmitData=pd.DataFrame(X_test,columns=predictorColumns)
AdmitData['ChanceOfAdmit']=y_test
AdmitData['PredictedChancesOfAdmit']=PreAdmit
rmseLin2=np.sqrt(mean_squared_error(y_test,PreAdmit))
accuracyLin2=r2_score(y_test,PreAdmit)

#Deceision tree
model=DecisionTreeRegressor(random_state=36,max_depth=100,min_samples_leaf=10)
model.fit(X_train,y_train)
PreAdmit=model.predict(X_test)
AdmitData=pd.DataFrame(X_test,columns=predictorColumns)
AdmitData['ChanceOfAdmit']=y_test
AdmitData['PredictedChancesOfAdmit']=PreAdmit
rmseDT2=np.sqrt(mean_squared_error(y_test,PreAdmit))
accuracyDT2=r2_score(y_test,PreAdmit)



#KNN
model=KNeighborsRegressor(n_neighbors=9,metric='euclidean')
model.fit(X_train,y_train)
PreAdmit=model.predict(X_test)
AdmitData=pd.DataFrame(X_test,columns=predictorColumns)
AdmitData['ChanceOfAdmit']=y_test
AdmitData['PredictedChancesOfAdmit']=PreAdmit
rmseKNN2=np.sqrt(mean_squared_error(y_test,PreAdmit))
accuracyKNN2=r2_score(y_test,PreAdmit)



#Nerual Network
model=MLPRegressor(hidden_layer_sizes=50,random_state=10)
model.fit(X_train,y_train)
PreAdmit=model.predict(X_test)
AdmitData=pd.DataFrame(X_test,columns=predictorColumns)
AdmitData['ChanceOfAdmit']=y_test
AdmitData['PredictedChancesOfAdmit']=PreAdmit
rmseNN2=np.sqrt(mean_squared_error(y_test,PreAdmit))
accuracyNN2=r2_score(y_test,PreAdmit)






y = np.array([rmse,rmse2,rmseLin,rmseLin2,rmseKNN,rmseKNN2,rmseNN,rmseNN2,rmseDT,rmseDT2])
x = ["RF","RF3","LR","LR3","KNN","KNN3","NN","NN3","DT","DT3"]
listbar=plt.bar(x,y)
listbar[1].set_color('r')
listbar[3].set_color('r')
listbar[5].set_color('r')
listbar[7].set_color('r')
listbar[9].set_color('r')
plt.title("Comparison of Regression Algorithms")
plt.xlabel("Regressor")
plt.ylabel("rmse")
plt.show()


y = np.array([accuracy,accuracy2,accuracyLin,accuracyLin2,accuracyKNN,accuracyKNN2,accuracyNN,accuracyNN2,accuracyDT,accuracyDT2])
x = ["RF","RF3","LR","LR3","KNN","KNN3","NN","NN3","DT","DT3"]
listbar=plt.bar(x,y)
listbar[1].set_color('r')
listbar[3].set_color('r')
listbar[5].set_color('r')
listbar[7].set_color('r')
listbar[9].set_color('r')
plt.title("Comparison of Regression Algorithms")
plt.xlabel("Regressor")
plt.ylabel("r2_score")
plt.show()