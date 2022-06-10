import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.linear_model import CoxPHSurvivalAnalysis
from lifelines import CoxPHFitter
import warnings
warnings.filterwarnings('ignore')
import json

df = pd.read_csv("survival.csv")

df.rename(columns={'Vibration - X':'VibrationX','Vibration  Y':'VibrationY','Vibration  Z':'VibrationZ'},inplace=True)

columns = ['Lifetime', 'Broken', 'Temperature', 'Voltage', 'Current', 'Humidity','VibrationX', 'VibrationY', 'VibrationZ']
data = df.loc[:,columns]

coxph = CoxPHFitter()
coxph.fit(data, duration_col='Lifetime', event_col='Broken')

def predict(a,c,d,e,f,g,h,i):
    data1 = pd.DataFrame({'Lifetime':[a],'Temperature':[c],'Voltage':[d],'Current':[e],'Humidity':[f],
                          'VibrationX':[g],'VibrationY':[h],'VibrationZ':[i]})
    pred_prob = coxph.predict_survival_function(data1,conditional_after=data1['Lifetime'])
    
    lifetime1 = []
    for i in pred_prob.columns:
        lifetime1.append(pred_prob.index[pred_prob[i]==pred_prob[i].quantile(0.5)][0])
        
    result1 = pd.DataFrame({'pred_lifetime':lifetime1})
    output1 = pd.merge(data1,result1,left_index=True, right_index=True)
    output1['RUL'] = output1.pred_lifetime - output1.Lifetime
    #print(output1['pred_lifetime'][0])
    #print(output1['RUL'][0])

import pickle

with open('RUL_Cox.pkl', 'wb') as files:
    pickle.dump(coxph, files)

with open('RUL_Cox.pkl' , 'rb') as f:
    RUL_Cox = pickle.load(f)

Bdt = pickle.load(open("RUL_Cox.pkl",'rb'))

def predict_RUL(input_data):

    a = int(input_data["Lifetime"])
    c = float(input_data["Temperature"])
    d = float(input_data["Voltage"])
    e = float(input_data["Current"])
    f = float(input_data["Humidity"])
    g = float(input_data["VibrationX"])
    h = float(input_data["VibrationY"])
    i = float(input_data["VibrationZ"])
        
        
        
    #get prediction
    #def predict1(a,b,c,d,e,f,g,h,i):
    data1 = pd.DataFrame({'Lifetime':[a],'Temperature':[c],'Voltage':[d],'Current':[e],'Humidity':[f],
                        'VibrationX':[g],'VibrationY':[h],'VibrationZ':[i]})
    pred_prob = coxph.predict_survival_function(data1,conditional_after=data1['Lifetime'])

    lifetime1 = []
    for i in pred_prob.columns:
        lifetime1.append(pred_prob.index[pred_prob[i]==pred_prob[i].quantile(0.5)][0])
    
    result1 = pd.DataFrame({'pred_lifetime':lifetime1})
    output1 = pd.merge(data1,result1,left_index=True, right_index=True)
    output1['RUL'] = output1.pred_lifetime - output1.Lifetime
    output2 = list(output1['RUL'])
    output = json.dumps(output2)
    return output