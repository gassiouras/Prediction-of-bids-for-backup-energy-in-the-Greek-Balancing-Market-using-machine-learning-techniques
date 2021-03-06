# -*- coding: utf-8 -*-
"""Encoder_Decoder.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1-bckq-VkpLBHs9SdiF_KuXVaJtxg5UAl
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.layers import *
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from keras.callbacks import EarlyStopping
from numpy import *
from pandas import *
from math import sqrt

from google.colab import drive
drive.mount('/content/drive')

#if you want to predict variables for Up reserve energy set energy_type='Up'
#if you want to predict variables for Down reserve energy set energy_type='Down'
energy_type='Up'

#if you want to predict the quantity set var_type=0
#if you want to predict the price set var_type=1
var_type=0

if energy_type==True:
  df= pd.read_excel('/content/drive/MyDrive/BE/BE_names.xlsx')
else:
  df= pd.read_excel('/content/drive/MyDrive/BE/Actual_Units_Down.xlsx')
Date= pd.read_excel('/content/drive/MyDrive/BE/DateDim.xlsx')

Date['Day'] = pd.to_datetime(Date['Day'])
Date = Date.set_index('Day')
Date = Date.loc['2021-02-13':'2021-04-16']

date1 = Date.loc[:, 'SinWeekDay':'CosHour']



entityname=df.names.values
entityname1 = []
entityname1.append('PROTERGIA_CC')
for i in entityname1:
  print(i)
  vars()[i] = pd.read_excel('/content/drive/MyDrive/BE/BE_'+str(energy_type)+'_New_New/%s.xlsx' % i)
  vars()[i] = vars()[i].set_index('Date')
  vars()[i] = vars()[i].fillna(0)

entityname1 = []
entityname1.append('KASTRAKI')

def quantity_values(entity, max_steps):
  quantity = entity.loc[:, 'Step1Quantity':'Step'+str(max_steps)+'Quantity']
  for i in range(1, max_steps+1):
    
    column_values = quantity['Step'+str(i)+'Quantity'].values
    stepval = 'Step'+str(i)+'Val'
    vars()[stepval] =  np.unique(column_values)
    vars()[stepval] = pd.DataFrame(vars()[stepval])
    if i ==1:
      val = vars()[stepval].copy()
    else:
      val = pd.concat([val, vars()[stepval]], axis=1)
  
  val= val.fillna(0)
  return val

def classification(value, x, max_quantity):
  if x<0:
    clas = 0
  elif x>max_quantity:
    clas = max_quantity
  else:
    d = []
    for i in range(len(value)):
      d.append(abs(x-value.iat[i]))
    min = d[0]
    clas = value[0]
    for j in range(1, len(d)):
      if d[j]<min:
        min = d[j]
        clas = value.iat[j]
  return int(clas)

def classification3(value, x, max_quantity):
  if x<0:
    clas = 0
  else:
    clas = x
  return float(clas)

def accuracy_score_manual(pred, real):
  k=0
  for i in range(0, len(pred)):
    if pred.iat[i] == real.iat[i]:
      k+=1
  return k*100/len(pred)

def split_series(series, n_past, n_future, n_features_out):
  #
  # n_past ==> no of past observations
  #
  # n_future ==> no of future observations 
  #
  X, y = list(), list()
  for window_start in range(len(series)):
    past_end = window_start + n_past
    future_end = past_end + n_future
    if future_end > len(series):
      break
    # slicing the past and future parts of the window
    past, future = series[window_start:past_end, :], series[past_end:future_end, :n_features_out]
    X.append(past)
    y.append(future)
  return np.array(X), np.array(y)

scl = MinMaxScaler()

column_name = []
for k in range(0, 336):
  column_name.append('predicted')
  column_name.append('classified')
  column_name.append('real')

error_index=[]
for i in entityname:
  error_index.append(''+str(i)+'')
  error_index.append('')
error_index.append('mean real')
error_index.append('mean clas')

error_name = []
error_name.append('Type of pred')
for k in range(1,11):
  error_name.append('MAE')
  error_name.append('RMSE')
error_name.append('Accuracy')

g=0
tot_error = pd.DataFrame(index=np.arange(len(error_index)), columns=np.arange(22))
for p in range(0, int(len(tot_error)/2)):
  tot_error.iat[2*p,0] = 'orig'
  tot_error.iat[2*p+1,0] = 'class'
tot_pred = 0
weeks_of_pred = 7
if var_type=0:
  col_num = 0 #col num=0 for quantity and clo_num=10 for prices
  diagram_names = 'Quantity'
  save_names = 'Quantity'
else:
  col_num = 10 #col num=0 for quantity and clo_num=10 for prices
  diagram_names = 'Price'
  save_names = 'Price'

for i in entityname1:
  entity = vars()[i].loc[:, :]
  final_step = max(entity.NumOfSteps)
  mae = pd.DataFrame(index=np.arange(2*weeks_of_pred+2), columns=np.arange(2*final_step+3))
  mae_col_name = []
  mae_col_name.append('Date')
  mae_col_name.append('')
  for j in range(1, final_step+1):
    mae_col_name.append('MAE')
    mae_col_name.append('RMSE')
  mae_col_name.append('Accuracy')
  mae.columns = mae_col_name
  for e in range(0, 7):
      
    entity = entity.loc['2021-02-13':'2021-07-14']
    variable= entity.loc[:, 'VariableCost']
    availability = entity.loc[:, 'Availability']
      
    min_step = min(entity.NumOfSteps)
    final_step = max(entity.NumOfSteps)
    entity1 = entity.iloc[48*e:len(entity)-(48*(6-e)), col_num+3:final_step+col_num+3]
    max_list = entity1.max(axis=0).values
    min_list = entity1.min(axis=0).values
    max_quantity = entity1.iloc[0:1, :].sum(axis=1)
    max_quantity = max_quantity.values
      
      
      #dataset1 = dataset1.replace(np.nan, 0)
      #print(dataset1.values.shape)
      #print(entity1)
    if entity.iat[0, 3] >0:
      tot_pred +=1
      dataset = entity1.copy()
      #dataset = pd.concat([entity1, date1, variable, availability], axis=1)
      extra_val=0
      dataset = dataset.replace(np.nan, 0)
        #print(dataset1.values.shape)
        #print(entity1)
      
      for k in range(1, dataset.shape[0], 7*48):
        if abs(dataset.shape[0]*0.6-k) < abs(dataset.shape[0]*0.6-(k-1)):
          split = k-1
      train, test = dataset[0:split], dataset[split:]
        
      scalers={}
      for k in train.columns:
        scaler = MinMaxScaler()
        s_s = scaler.fit_transform(train[k].values.reshape(-1,1))
        s_s=np.reshape(s_s,len(s_s))
        scalers['scaler_'+k] = scaler
        train[k]=s_s

      for k in train.columns:
        scaler = scalers['scaler_'+k]
        s_s = scaler.transform(test[k].values.reshape(-1,1))
        s_s=np.reshape(s_s,len(s_s))
        scalers['scaler_'+k] = scaler
        test[k]=s_s
      test_model  = test[:-672]
      test_pred = test[-672:]
      test_out = test_pred[-336:]


      n_past = 1*7*48
      n_future = 1*7*48 
      n_features_in = final_step+extra_val
      n_features_out = final_step
        
      X_train, y_train = split_series(train.values,n_past, n_future, n_features_out)
      X_train = X_train.reshape((X_train.shape[0], X_train.shape[1],n_features_in))
      y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], n_features_out))
      X_test_model, y_test_model = split_series(test_model.values,n_past, n_future, n_features_out)
      X_test_model = X_test_model.reshape((X_test_model.shape[0], X_test_model.shape[1],n_features_in))
      y_test_model = y_test_model.reshape((y_test_model.shape[0], y_test_model.shape[1], n_features_out))
      X_test_pred, y_test_pred = split_series(test_pred.values,n_past, n_future, n_features_out)
      X_test_pred = X_test_pred.reshape((X_test_pred.shape[0], X_test_pred.shape[1],n_features_in))
      y_test_pred = y_test_pred.reshape((y_test_pred.shape[0], y_test_pred.shape[1], n_features_out))

      encoder_inputs = tf.keras.layers.Input(shape=(n_past, n_features_in))
      encoder_l1 = tf.keras.layers.LSTM(10, return_state=True)
      encoder_outputs1 = encoder_l1(encoder_inputs)

      encoder_states1 = encoder_outputs1[1:]

       #
      decoder_inputs = tf.keras.layers.RepeatVector(n_future)(encoder_outputs1[0])

        #
      decoder_l1 = tf.keras.layers.LSTM(10, return_sequences=True)(decoder_inputs,initial_state = encoder_states1)
      decoder_outputs1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features_out))(decoder_l1)

      #
      model_e1d1 = tf.keras.models.Model(encoder_inputs,decoder_outputs1)

        #
        #model_e1d1.summary()

        #model
      #encoder_inputs = tf.keras.layers.Input(shape=(n_past, n_features_in))
      #encoder_l1 = tf.keras.layers.LSTM(100,return_sequences = True, return_state=True)
      #encoder_outputs1 = encoder_l1(encoder_inputs)
      #encoder_states1 = encoder_outputs1[1:]
      #encoder_l2 = tf.keras.layers.LSTM(100, return_state=True)
      #encoder_outputs2 = encoder_l2(encoder_outputs1[0])
      #encoder_states2 = encoder_outputs2[1:]
        #
      #decoder_inputs = tf.keras.layers.RepeatVector(n_future)(encoder_outputs2[0])
        #
      #decoder_l1 = tf.keras.layers.LSTM(100, return_sequences=True)(decoder_inputs,initial_state = encoder_states1)
      #decoder_l2 = tf.keras.layers.LSTM(100, return_sequences=True)(decoder_l1,initial_state = encoder_states2)
      #decoder_outputs2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features_out))(decoder_l2)
        #
      #model_e1d1 = tf.keras.models.Model(encoder_inputs,decoder_outputs2)
        #
        #model_e2d2.summary()
      print(model_e1d1.summary())
      reduce_lr = tf.keras.callbacks.LearningRateScheduler(lambda x: 1e-3 * 0.90 ** x)
      model_e1d1.compile(optimizer='adam', loss='mse')

      history = model_e1d1.fit(X_train,y_train,epochs=100, validation_data=(X_test_model, y_test_model), 
                              callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)], shuffle=False)
        
      pred_e1d1=model_e1d1.predict(X_test_pred)
        
        
      for j in range(0, pred_e1d1.shape[1]):
        for k in range(0, pred_e1d1.shape[2]):
          pred_e1d1[0,j,k] = pred_e1d1[0,j,k]*(max_list[k]-min_list[k])+min_list[k]
          y_test_pred[0,j,k] = y_test_pred[0,j,k]*(max_list[k]-min_list[k])+min_list[k]


        
        #prediction1 = pred_e1d1.copy()
        #value = quantity_values(entity, final_step)
        #if len(value)<7:
        #  for j in range(0, pred_e1d1.shape[1] ):
        #    prediction1[0,j,:] = classification(value, pred_e1d1[0,j,:],  max_list, max_quantity, min_step)

      a = pd.DataFrame(index=np.arange(pred_e1d1.shape[1]), columns=np.arange(3*final_step+3))
        
      value = quantity_values(entity, final_step)
      name = []
      for j in range(0, final_step):
        name.append('Step'+str(j+1)+'Predicted')
        name.append('Step'+str(j+1)+'Classified')
        name.append('Step'+str(j+1)+'Real')
      name.append('NumOfSTeps_pred')
      name.append('NumOfSTeps_clas')
      name.append('NumOfSteps_real')
      a.index = test_out.index
      a.columns = name
      for k in range(pred_e1d1.shape[1]):
        pr = 0
        cl_v = 0
        real_v = 0
        for j in range(pred_e1d1.shape[2]):
          a.iat[k,3*j] = pred_e1d1[0,k,j]
          if pred_e1d1[0,k,j]>0:
            pr += 1
            #a.iat[k,3*j+1] = prediction1[0,k,j]
          a.iat[k,3*j+1] = classification3(value.iloc[:,j], pred_e1d1[0,k,j], max_quantity)
          if a.iat[k,3*j+1]>0:
            cl_v +=1
          a.iat[k,3*j+2] = y_test_pred[0,k,j]
          if y_test_pred[0,k,j]>0:
            real_v +=1
        a.iat[k,3*pred_e1d1.shape[2]] = pr
        a.iat[k,3*pred_e1d1.shape[2]+1] = cl_v
        a.iat[k,3*pred_e1d1.shape[2]+2] = real_v
      a.to_excel('/content/drive/MyDrive/BE/BE_'+str(energy_type)+'n_New/EncoderLSTM/7dayspred/'+str(save_names)+'/predictions/'+str(i)+' prediction at '+str(e+1)+'day.xlsx')
        
      fig = plt
      a['Step1Real'].plot(figsize=(30,18), fontsize=14)
      a['Step1Predicted'].plot(figsize=(30,18), fontsize=20)
      a['Step1Classified'].plot(figsize=(30,18), fontsize=20)
      plt.title(''+str(i)+'  Step1 '+str(diagram_names)+'', fontsize=30)
      plt.xlabel("Date", fontsize=30)
      plt.ylabel(''+str(diagram_names)+'', fontsize=30)
      plt.xticks(fontsize=30)
      plt.yticks(fontsize=30)
      plt.legend(prop={'size':30})
        #plt.show()
      fig.savefig('/content/drive/MyDrive/BE/BE_'+str(energy_type)+'_New/EncoderLSTM/7dayspred/'+str(save_names)+'/diagram/'+str(i)+' prediction at '+str(e+1)+'day.png')
      #if e==6:
        #fig.savefig('/content/drive/MyDrive/BE/BE_Down_New/EncoderLSTM/7dayspred/diagram_final/'+str(i)+'.png')
      plt.close()




      index_list = a.index.tolist()
      first_day = index_list[0].day
      first_day_month = index_list[0].month
      last_day = index_list[len(index_list)-1].day
      last_day_month = index_list[len(index_list)-1].month
      date = '{}/{}-{}/{}'.format(first_day, first_day_month, last_day, last_day_month)
      mae.iat[2*e,0] = date
      mae.iat[2*e+1,0] = ''
        #print(a.iloc[:, 5])
        
      for p in range(0, 7):
        mae.iat[2*p, 1] = 'Orig'
        mae.iat[2*p+1, 1] = 'Class'
      
      for k in range(0, final_step): 
        mae.iat[2*e, 2*k+2] = mean_absolute_error(a.iloc[:, 3*k+2], a.iloc[:, 3*k])
        mae.iat[2*e, 2*k+3] = sqrt(mean_squared_error(a.iloc[:, 3*k+2], a.iloc[:, 3*k]))
        mae.iat[2*e+1, 2*k+2] = mean_absolute_error(a.iloc[:, 3*k+2], a.iloc[:, 3*k+1])
        mae.iat[2*e+1, 2*k+3] = sqrt(mean_squared_error(a.iloc[:, 3*k+2], a.iloc[:, 3*k+1]))
      mae.iat[2*e, 2*final_step+2] = accuracy_score_manual(a.iloc[:,3*final_step], a.iloc[:, 3*final_step+2])
      mae.iat[2*e+1, 2*final_step+2] = accuracy_score_manual(a.iloc[:,3*final_step+1], a.iloc[:, 3*final_step+2])


  mae.iat[2*weeks_of_pred,0] = 'mean real error'
  mae.iat[2*weeks_of_pred+1,0] = 'mean clas error'
  mae.columns = mae_col_name
  mae =  mae.set_index('Date')
  for k in range(1, len(mae.columns)):
    sum1=0
    sum2=0

    for p in range(0, weeks_of_pred): 
      sum1 += mae.iat[2*p,k]
      sum2 = sum2 + mae.iat[2*p+1,k]
        
    mae.iat[2*weeks_of_pred, k] = sum1/(weeks_of_pred)
    mae.iat[2*weeks_of_pred+1, k] = sum2/(weeks_of_pred)

    if k ==len(mae.columns)-1:
      tot_error.iat[g, 21] = mae.iat[2*weeks_of_pred, k]
      tot_error.iat[g+1, 21] = mae.iat[2*weeks_of_pred+1, k]
    else:
      tot_error.iat[g, k] = mae.iat[2*weeks_of_pred, k]
      tot_error.iat[g+1, k] = mae.iat[2*weeks_of_pred+1, k]
  mae.to_excel('/content/drive/MyDrive/BE/BE_'+str(energy_type)+'_New/EncoderLSTM/7dayspred/'+str(save_names)+'/mape/'+str(i)+'.xlsx')

  print (g, i)
  g = g+2

tot_error = tot_error.fillna(0)
for j in range(1, len(tot_error.columns)):

  sum1 =0
  sum2 = 0
  for p in range(0, int(len(tot_error)/2)-1):
    sum1 += tot_error.iat[2*p, j]
    sum2 += tot_error.iat[2*p+1, j]
    
  tot_error.iat[len(tot_error)-2, j] = sum1/tot_pred
  tot_error.iat[len(tot_error)-1, j] = sum2/tot_pred



tot_error.columns = error_name
tot_error.index = error_index
tot_error.to_excel('/content/drive/MyDrive/BE/BE_'+str(energy_type)+'_New/EncoderLSTM/7dayspred/'+str(save_names)+'/En-De .xlsx')