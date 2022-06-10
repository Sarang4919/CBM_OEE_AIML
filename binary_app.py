import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.tree import DecisionTreeClassifier
import json

df = pd.read_csv('survival.csv')

df.rename(columns={'Vibration - X':'VibrationX','Vibration  Y':'VibrationY','Vibration  Z':'VibrationZ'},inplace=True)
df1 = df.drop(['Machine ID','Responsible Failure Cause','Failure Model'],axis=1)

df1_target = df1['Broken']
df1_feature = df1.drop('Broken',axis=1)

df1_num = df1_feature.select_dtypes(include=np.number)

df1_cat = df1_feature.select_dtypes(include=object)

LE = LabelEncoder()

df1_cat['Machine_Name'] = LE.fit_transform(df1_cat['Machine Name'])

dummy_var = df1_cat.drop(['Machine Name'],axis=1).copy()

X = pd.concat([df1_num,dummy_var],axis=1)


X_train,X_test,y_train,y_test = train_test_split(X,df1_target,test_size=0.3,random_state=10)

decision_tree_classification = DecisionTreeClassifier(criterion='entropy',random_state=10)

decision_tree = decision_tree_classification.fit(X_train,y_train)

y_pred_decision_tree = decision_tree.predict(X_test)

y_pred_probab = np.round(decision_tree.predict_proba(X_test)[:,1],2)

import pickle

with open('Binary_DT.pkl', 'wb') as files:
    pickle.dump(decision_tree, files)

with open('Binary_DT.pkl' , 'rb') as f:
    Bdt = pickle.load(f)

Bdt = pickle.load(open("Binary_DT.pkl",'rb'))

def predict_binary(input_data):

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
    prediction = np.round(Bdt.predict_proba(input_cols)[:,1],2)
    prediction1 = (prediction)
    prediction2 = list(prediction1)
    output = json.dumps(prediction2)

    return output
