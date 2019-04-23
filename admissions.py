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

def get_plt_params():
    params = {'legend.fontsize': 'x-large',
              'figure.figsize' : (18, 8),
              'axes.labelsize' : 'x-large',
              'axes.titlesize' : 'x-large',
              'xtick.labelsize': 'x-large',
              'ytick.labelsize': 'x-large',
              'font.size'      :  10}
    return params

columns = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA', 'Research']
target = 'Chance of Admit'

df = pd.read_csv('../data/Admission_Predict_Ver1.1.csv')
'''
fig,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(df.corr(), ax=ax, annot=True, linewidths=0.05, fmt= '.2f',cmap="magma")
plt.show()
'''
#print(df.head())
#print(df.corr())
#print(df.isnull().sum())
'''
fig = plt.figure(figsize=(18,9))
params = get_plt_params()
plt.rcParams.update(params)
fig.subplots_adjust(hspace=0.5)
for i in range(len(columns)-1) :
    plt.subplot(2,3,i+1)
    sns.lineplot(x=columns[i],y=target,data=df)
plt.tight_layout()
plt.plot()

plt.show()
'''
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
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=36)

#Random foreast
'''
trees=np.arange(5,200,1)
max_depth=np.arange(5,60,5)
res=[]
res2=[]
for num in trees :
    clf = RandomForestRegressor(n_estimators=num, criterion='mse',min_samples_split=50)
    RF = clf.fit(X_train, y_train)
    PreAdmitRF = RF.predict(X_test)
    accuracyRF = r2_score(y_test, PreAdmitRF)
    res.append(accuracyRF)
    
    PreAdmitRF2 = RF.predict(X_train)
    accuracyRF2 = r2_score(y_train, PreAdmitRF2)
    res2.append(accuracyRF2)
    
plt.xlabel("number of trees")
plt.ylabel("r2_score")
plt.plot(trees,res,'ro')
plt.axes([5,200,0,1])
plt.show()
'''
'''
max_depth=np.arange(5,60,5)
res=[]
for num in max_depth :
    clf = RandomForestRegressor(n_estimators=100, criterion='mse',max_depth=num,random_state=10)
    RF = clf.fit(X_train, y_train)
    PreAdmitRF = RF.predict(X_test)
    accuracyRF = r2_score(y_test, PreAdmitRF)
    res.append(accuracyRF)
plt.xlabel("max_depth")
plt.ylabel("r2_score")
plt.plot(max_depth,res,'ro')
plt.axes([0,50,0,1])
plt.show()
'''


clf=RandomForestRegressor(n_estimators=100,max_depth=45,criterion='mse',min_samples_leaf=12,random_state=36)
RF=clf.fit(X_train,y_train)
PreAdmitRF=RF.predict(X_test)
AdmitData=pd.DataFrame(X_test,columns=predictorColumns)
AdmitData['ChanceOfAdmit']=y_test
AdmitData['PredictedChancesOfAdmit']=PreAdmitRF
rmseRF=np.sqrt(mean_squared_error(y_test,PreAdmitRF))
accuracyRF=r2_score(y_test,PreAdmitRF)
PreAdmitRF2=RF.predict(X_train)
accuracyRF2=r2_score(y_train,PreAdmitRF2)
print("RF test: "+str(accuracyRF))
print("RF training: "+str(accuracyRF2))


#Linear Regression
model=LinearRegression()
model.fit(X_train,y_train)
PreAdmitLR=model.predict(X_test)
AdmitData=pd.DataFrame(X_test,columns=predictorColumns)
AdmitData['ChanceOfAdmit']=y_test
AdmitData['PredictedChancesOfAdmit']=PreAdmitLR
rmseLin=np.sqrt(mean_squared_error(y_test,PreAdmitLR))
accuracyLin=r2_score(y_test,PreAdmitLR)
PreAdmitLR2=model.predict(X_train)
accuracyLin2=r2_score(y_train,PreAdmitLR2)
print("test :"+str(accuracyLin))
print("train :"+str(accuracyLin2))

#Deceision tree
model=DecisionTreeRegressor(random_state=36,max_depth=100,min_samples_leaf=10)
model.fit(X_train,y_train)
PreAdmitDT=model.predict(X_test)
AdmitData=pd.DataFrame(X_test,columns=predictorColumns)
AdmitData['ChanceOfAdmit']=y_test
AdmitData['PredictedChancesOfAdmit']=PreAdmitDT
rmseDT=np.sqrt(mean_squared_error(y_test,PreAdmitDT))
accuracyDT=r2_score(y_test,PreAdmitDT)
PreAdmitDT2=model.predict(X_train)
accuracyDT2=r2_score(y_train,PreAdmitDT2)
print("test : "+str(accuracyDT))
print("trainging : "+str(accuracyDT2))

#KNN
model=KNeighborsRegressor(n_neighbors=9,metric='euclidean')
model.fit(X_train,y_train)
PreAdmitKNN=model.predict(X_test)
AdmitData=pd.DataFrame(X_test,columns=predictorColumns)
AdmitData['ChanceOfAdmit']=y_test
AdmitData['PredictedChancesOfAdmit']=PreAdmitKNN
rmseKNN=np.sqrt(mean_squared_error(y_test,PreAdmitKNN))
accuracyKNN=r2_score(y_test,PreAdmitKNN)
PreAdmitKNN2=model.predict(X_train)
accuracyKNN2=r2_score(y_train,PreAdmitKNN2)
print("test : "+str(accuracyKNN))
print("training : "+str(accuracyKNN2))

#Nerual Network
model=MLPRegressor(hidden_layer_sizes=50,random_state=10)
model.fit(X_train,y_train)
PreAdmitNN=model.predict(X_test)
AdmitData=pd.DataFrame(X_test,columns=predictorColumns)
AdmitData['ChanceOfAdmit']=y_test
AdmitData['PredictedChancesOfAdmit']=PreAdmitNN
rmseNN=np.sqrt(mean_squared_error(y_test,PreAdmitNN))
accuracyNN=r2_score(y_test,PreAdmitNN)
PreAdmitNN2=model.predict(X_train)
accuracyNN2=r2_score(y_train,PreAdmitNN2)
print("test : "+str(accuracyNN))
print("training : "+str(accuracyNN2))


y = np.array([rmseRF,rmseLin,rmseKNN,rmseNN,rmseDT])
x = ["RandomForestReg.","LinearRegression","KNN","Neural Network","Deceision tree"]
plt.bar(x,y)
plt.title("Comparison of Regression Algorithms")
plt.xlabel("Regressor")
plt.ylabel("rmse")
plt.show()



y = np.array([accuracyRF,accuracyLin,accuracyKNN,accuracyNN,accuracyDT])
x = ["RandomForestReg.","LinearRegression","KNN","Neural Network","Deceision tree"]
plt.bar(x,y)
plt.title("Comparison of Regression Algorithms")
plt.xlabel("Regressor")
plt.ylabel("r2_score")
plt.show()


red = plt.scatter(np.arange(0,150,5),PreAdmitLR[0:150:5],color = "red")
green = plt.scatter(np.arange(0,150,5),PreAdmitRF[0:150:5],color = "green")
blue = plt.scatter(np.arange(0,150,5),PreAdmitDT[0:150:5],color = "blue")
black = plt.scatter(np.arange(0,150,5),PreAdmitKNN[0:150:5],color = "black")
yellow = plt.scatter(np.arange(0,150,5),PreAdmitNN[0:150:5],color = "yellow")
pink = plt.scatter(np.arange(0,150,5),y_test[0:150:5],color = "pink")
plt.title("Comparison of Regression Algorithms")
plt.xlabel("Index of Candidate")
plt.ylabel("Chance of Admit")
plt.legend((red,green,blue,black,yellow,pink),('LR', 'RF', 'DT', 'KNN','NN','Actual'))
plt.show()
