import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')
import json

df = pd.read_csv('survival.csv')

df.rename(columns={'Vibration - X':'VibrationX','Vibration  Y':'VibrationY','Vibration  Z':'VibrationZ'},inplace=True)

df1 = df.drop(['Machine ID','Responsible Failure Cause','Broken'],axis=1)

df1_cat = df1[['Machine Name','Failure Model']]

LE = LabelEncoder()

df1_cat['Machine_Name'] = LE.fit_transform(df1_cat['Machine Name'])

df1_cat['Failure_Model'] = LE.fit_transform(df1['Failure Model'])

dummy_var = df1_cat[['Machine_Name','Failure_Model']]

df1_num = df1.select_dtypes(include=np.number)

X = pd.concat([df1_num,dummy_var],axis=1)

df1_target = X['Failure_Model']
df1_feature = X.drop('Failure_Model',axis=1)

X_train,X_test,y_train,y_test = train_test_split(df1_feature,df1_target,test_size=0.3,random_state=10)

decision_tree_classification = DecisionTreeClassifier(criterion='entropy',random_state=10)

decision_tree = decision_tree_classification.fit(X_train,y_train)

y_pred_decision_tree = decision_tree.predict(X_test)

import pickle

with open('Multiclass_DT.pkl', 'wb') as files:
    pickle.dump(decision_tree, files)

with open('Multiclass_DT.pkl' , 'rb') as f:
    Bdt = pickle.load(f)

Bdt = pickle.load(open("Multiclass_DT.pkl",'rb'))

def predict_multiclass(input_data):
    Lifetime = int(input_data["Lifetime"])
    Temperature = float(input_data["Temperature"])
    Voltage = float(input_data["Voltage"])
    Current = float(input_data["Current"])
    Humidity = float(input_data["Humidity"])
    VibrationX = float(input_data["VibrationX"])
    VibrationY = float(input_data["VibrationY"])
    VibrationZ = float(input_data["VibrationZ"])
    Machine_Name = str(input_data['Machine_Name'])
    
    #get prediction
    input_cols = [[Lifetime,Temperature, Voltage, Current, Humidity,VibrationX, VibrationY, VibrationZ,Machine_Name]]
    prediction = Bdt.predict(input_cols)
    prediction1 = pd.DataFrame(prediction, columns=['predictions'])
    prediction1['predictions'] = prediction1['predictions'].astype(int)
    prediction1["predictions"] = prediction1['predictions'].map({0:'Electrical Model', 1:'Healthy State',2:'Lifetime Model',3:'Thermal Model',4:'Vibration Model'})
    output1 = list(prediction1['predictions'])
    output = json.dumps(output1)
    return output
