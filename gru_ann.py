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
from sklearn.metrics import *
from math import sqrt

from google.colab import drive
drive.mount('/content/drive')

#read the units name depending on which type of energy we want to predict
def read_names(self, energy_type):
    self.energy_type = energy_type
    if energy_type==True:
        units_names = pd.read_excel('/content/drive/MyDrive/BE/BE_names.xlsx')
    else:
        units_names = pd.read_excel('/content/drive/MyDrive/BE/Actual_Units_Down.xlsx')
    Date= pd.read_excel('/content/drive/MyDrive/BE/DateDim.xlsx')
    return (units_names, Date)

def percentage_error(actual, predicted):
  res = list()
  for j in range(actual.shape[0]):
    if actual[j] != 0:
      res.append((actual[j] - predicted[j]) / actual[j])
    elif np.mean(actual)!=0:
      res.append(predicted[j] / np.mean(actual))
    else: 
      res.append(0)
  return res

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

def split_sequences(data, lookback, lookforward, max_steps):
	X, y = list(), list()
	for i in range(len(data)-lookback):
		end_ix = i + lookback
		out_end_ix = end_ix + lookforward
		if out_end_ix > len(data):
			break
		seq_x, seq_y = data[i:end_ix, 0:data.shape[1]], data[end_ix:out_end_ix, 0:max_steps]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)

def accuracy_score_manual(pred, real):
  k=0
  for i in range(0, len(pred)):
    if pred.iat[i] == real.iat[i]:
      k+=1
  return k*100/len(pred)

def classify_steps(max_steps, steps):
  k = pd.DataFrame(index=np.arange(len(steps)), columns=np.arange(max_steps))
  for i in range(0, len(steps)):

    for j in range(1, max_steps+1):
      if steps.iat[i] == j:
        k.iat[i,j-1] = 1
  k = k.fillna(0)
  return k


energy_type = ''
while not (energy_type.isalpha() and energy_type=='Up' or energy_type==('Down')):
    energy_type = input('Enter the type of energy you want (Up or Down)')


variable_type = ''
while not (variable_type.isalpha() and variable_type=='Quantity' or variable_type==('Price')):
    variable_type = input('Enter the type of variable you want to predict (Quantity or Price)')

var_type = 0
if variable_type=='Price':
    var_type = 1

df, Date = read_names(energy_type)
entityname=df.names.values

for i in entityname:
  vars()[i] = pd.read_excel('/content/drive/MyDrive/BE/BE_'+str(energy_type)+'_New/%s.xlsx' % i)
  vars()[i] = vars()[i].set_index('Date')
  vars()[i] = vars()[i].fillna(0)






Date['Day'] = pd.to_datetime(Date['Day'])
Date = Date.set_index('Day')


#after 13/02/2021 the units changed their pollicy so earlier data aren't useful
Date = Date.loc['2021-02-13':'2021-07-14']
  
date1 = Date.loc[:, 'SinWeekDay':'CosHour']


scl = MinMaxScaler()

column_name = []
for k in range(0, 336):
  column_name.append('predicted')
  column_name.append('real')


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

k=5

g=0
tot_error = pd.DataFrame(index=np.arange(len(error_index)), columns=np.arange(22))
tot_pred = 0
weeks_of_pred = 7
for i in entityname:
  entity = vars()[i]
  entity = entity.loc['2021-02-13':'2021-07-14']
  variable= entity.loc[:, 'VariableCost']
  availability = entity.loc[:, 'Availability']
  min_step = min(entity.NumOfSteps)
  final_step = max(entity.NumOfSteps)
  entity1 = entity.iloc[:, 3:3+final_step]
  max_list = entity1.max(axis=0).values
  min_list = entity1.min(axis=0).values
  max_quantity = entity1.iloc[0:1, :].sum(axis=1)
  max_quantity = max_quantity.values
  if var_type=0:
    col_num = 0 #col num=0 for quantity and clo_num=10 for prices
    diagram_names = 'Quantity'
    save_names = 'Quantity'
  else:
    col_num = 10 #col num=0 for quantity and clo_num=10 for prices
    diagram_names = 'Price'
    save_names = 'Price'
  max_steps=final_step
  if max_steps > 0:
    tot_pred += 1


    tot_pred += 1
    index_list = []
    mae = pd.DataFrame(index=np.arange(2*weeks_of_pred+2), columns=np.arange(2*final_step+3))
    real = entity['Step1'+str(diagram_names)+'']
    for t in range(0, 7):
      if i in entityname2:
        a = pd.read_excel('/content/drive/MyDrive/BE/BE_'+str(energy_type)+'_New/Classification/predictions/'+str(i)+'at'+str(t)+'day.xlsx')
        a = a.set_index('Date')
      else:

        a = vars()[i].loc['07-'+str(t+2)+'-2021':'07-'+str(t+8)+'-2021']

     # a = pd.DataFrame(index=np.arange(pred.shape[0]), columns=np.arange(2))
      
            
      #row=0
            
          
      #for j in range(0, pred.shape[1]):
       # a.iat[row, 0]= pred[48*t+47,j]
        #a.iat[row, 1] = real[48*t+47,j]
        #row = row+1
      #a = a.drop(a.index[len(a)-1]) 
          #a.columns = name
      #d = entity.iloc[(len(entity)-336-(6-t)*48):(len(entity)-(6-t)*48), :]
      
      #a.index = d.index
      #a.columns = ['Predicted', 'Real']
      #a.to_excel('/content/drive/MyDrive/BE/BE_Up_New/GRU-ANN/Classification/predictions/'+str(i)+'at'+str(t)+'day.xlsx')
      

      #ANN
      dataset2 = entity.iloc[:len(entity)-336-(6-t)*48, :]
      dataset3 = entity.iloc[len(entity)-336-(6-t)*48:(len(entity)-(6-t)*48), :]
      steps1  = dataset2.NumOfSteps
      if i in entityname2:
        steps2 = a.Predicted
      else:
        steps2 = a.NumOfSteps

      dataset2 = entity.iloc[:len(entity)-336-(6-t)*48, 3:3+max_steps]
      dataset3 = entity.iloc[len(entity)-336-(6-t)*48:(len(entity)-(6-t)*48), 3:3+max_steps]
      dataset_f1  = pd.concat([dataset2, dataset3])
      dataset_f2 = pd.concat([steps1, steps2])

      dataset_f3 = pd.concat([dataset_f1, dataset_f2], axis=1)
      dataset_f3 = dataset_f3.rename(columns = {dataset_f3.columns[max_steps]:'NumOfSteps'})
      
      dataset_final = dataset_f3
      
      cl2 = dataset_final.values.reshape(dataset_final.shape[0],len(dataset_final.columns))
      
      cl2 = scl.fit_transform(cl2)
      train_x = cl2[:int(len(cl2)*0.7), max_steps:]
      train_y = cl2[:int(len(cl2)*0.7), :max_steps]
      test_x = cl2[int(len(cl2)*0.7):len(cl2)-336, max_steps:]
      test_y = cl2[int(len(cl2)*0.7):len(cl2)-336, :max_steps]
      pred_x = cl2[len(cl2)-336:, max_steps:]
      pred_y = cl2[len(cl2)-336:, :max_steps]
      print(train_x.shape[1])
      #print(train_x.shape, train_y.shape, test_x.shape, test_x.shape, pred_x.shape, pred_y.shape)
      model2 = Sequential()
      model2.add(Dense(100, input_dim=train_x.shape[1]))
      model2.add(Dropout(0.05))
      #model2.add(Dense(80))
      #model2.add(Dropout(0.05))
      #model2.add(Dense(80))
      model2.add(Dense(train_y.shape[1]))
      model2.compile(loss='mse', optimizer='adam')
      print(model2.summary())
      history2 = model2.fit(train_x, train_y, epochs=100, validation_data=(test_x, test_y), 
                      callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)], shuffle=False)
      predictions = model2.predict(pred_x)
      
      
      prediction = predictions.copy()
      real_val = pred_y.copy()
      table2 = pd.DataFrame(index=np.arange(prediction.shape[0]), columns=np.arange(2*max_steps+2))

      for k in range(0,predictions.shape[0]):
        for j in range(0,predictions.shape[1]):
            prediction[k,j] = prediction[k,j]*(max_list[j]-min_list[j])+min_list[j]
            real_val[k,j] = real_val[k,j]*(max_list[j]-min_list[j])+min_list[j]
      prediction1 = prediction.copy()
      value = quantity_values(entity, max_steps)
      #if len(value)<5:
      #  for z in range(0, prediction.shape[0]):
      #      prediction1[z,:] = classification(value, prediction[z,:], max_list, max_quantity, min_step)

      
        
      
      table2 = pd.DataFrame(index=np.arange(prediction.shape[0]), columns=np.arange(3*final_step+3))

      
      
      for k in range(0, prediction.shape[0]):
        for j in range(0,prediction.shape[1]):
          table2.iat[k,3*j] = prediction[k,j]
          table2.iat[k,3*j+1] = classification( value.iloc[:,j], prediction[k,j],max_quantity )
          table2.iat[k, 3*j+2] = real_val[k,j]
        table2.iat[k, 3*final_step] = a.iat[k,0]
        table2.iat[k, 3*final_step+1] = a.iat[k,1]
    
      for k in range(0, len(table2)):
        pr=0
        cp=0
        rv = 0
        for j in range(0, final_step):
          if table2.iat[k,3*j]>0:
            pr += 1
          if table2.iat[k, 3*j+1]>0:
            cp +=1
          if table2.iat[k,3*j+2]>0:
            rv += 1
        table2.iat[k,3*final_step] = pr
        table2.iat[k, 3*final_step+1] = cp
        table2.iat[k, 3*final_step+2] = rv  
      name=[]
      for j in range(0, final_step):
        name.append('Step'+str(j+1)+'Predicted')
        name.append('Step'+str(j+1)+'Classified')
        name.append('Step'+str(j+1)+'Real')
      name.append('NumOfSTeps_pred')
      name.append('NumOfSTeps_clas')
      name.append('NumOfSteps_real')
      mae_col_name = []
      mae_col_name.append('Date')
      mae_col_name.append('')
      for j in range(1, final_step+1):
        mae_col_name.append('MAE')
        mae_col_name.append('RMSE')
      mae_col_name.append('Accuracy')

      table2.columns = name
      d = entity.iloc[(len(entity)-336-(6-t)*48):(len(entity)-(6-t)*48), :]
      table2.index = d.index
      table2 = table2.fillna(0)
      table2.to_excel('/content/drive/MyDrive/BE/BE_'+str(energy_type)+'_New/GRU-ANN/'+str(save_names)+'/predictions/'+str(i)+'at'+str(t)+'day.xlsx')
      col = 1
      fig = plt
      table2['Step1Real'].plot(figsize=(30,18), fontsize=30)
      table2['Step1Predicted'].plot(figsize=(30,18), fontsize=30)
      table2['Step1Classified'].plot(figsize=(30,18), fontsize=30)
      plt.title(''+str(i)+'  Step1 '+str(diagram_names)+'', fontsize=30)
      plt.xlabel("Date", fontsize=30)
      plt.ylabel(''+str(diagram_names)+'', fontsize=30)
      plt.xticks(fontsize=30)
      plt.yticks(fontsize=30)
      plt.legend(prop={'size':30})
      #plt.show()
      fig.savefig('/content/drive/MyDrive/BE/BE_'+str(energy_type)+'_New/GRU-ANN/'+str(save_names)+'/diagrams/'+str(i)+ ' prediction at '+str(t)+'day.png')
      if t==6:
        fig.savefig('/content/drive/MyDrive/BE/BE_'+str(energy_type)+'_New/GRU-ANN/'+str(save_names)+'/diagram_final/'+str(i)+ '.png')
      plt.close()

      index_list = a.index.tolist()
      first_day = index_list[0].day
      first_day_month = index_list[0].month
      last_day = index_list[len(index_list)-1].day
      last_day_month = index_list[len(index_list)-1].month
      date = '{}/{}-{}/{}'.format(first_day, first_day_month, last_day, last_day_month)
      mae.iat[2*t,0] = date
      mae.iat[2*t+1,0] = ''
      for p in range(0, 7):
        mae.iat[2*p, 1] = 'Orig'
        mae.iat[2*p+1, 1] = 'Class'
      
      for k in range(0, final_step): 
        mae.iat[2*t, 2*k+2] = mean_absolute_error(table2.iloc[:, 3*k+2], table2.iloc[:, 3*k])
        mae.iat[2*t, 2*k+3] = sqrt(mean_squared_error(table2.iloc[:, 3*k+2], table2.iloc[:, 3*k]))
        mae.iat[2*t+1, 2*k+2] = mean_absolute_error(table2.iloc[:, 3*k+2], table2.iloc[:, 3*k+1])
        mae.iat[2*t+1, 2*k+3] = sqrt(mean_squared_error(table2.iloc[:, 3*k+2], table2.iloc[:, 3*k+1]))
      mae.iat[2*t, 2*final_step+2] = accuracy_score_manual(table2.iloc[:,3*final_step], table2.iloc[:, 3*final_step+2])
      mae.iat[2*t+1, 2*final_step+2] = accuracy_score_manual(table2.iloc[:,3*final_step+1], table2.iloc[:, 3*final_step+2])
    
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
    mae.to_excel('/content/drive/MyDrive/BE/BE_'+str(energy_type)+'_New/GRU-ANN/'+str(save_names)+'/error/'+str(i)+'.xlsx')
    #calculate the mae for each day of prediction
    

    
    #diagram
    
    
    #save every prediction for every 30min

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
tot_error.to_excel('/content/drive/MyDrive/BE/BE_'+str(energy_type)+'_New/GRU-ANN/'+str(save_names)+'/only offers 4 layer.xlsx')

