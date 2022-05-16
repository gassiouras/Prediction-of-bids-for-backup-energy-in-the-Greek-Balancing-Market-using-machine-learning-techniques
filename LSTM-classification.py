import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from keras.callbacks import EarlyStopping

from sklearn.metrics import *

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

#count the quantity of energy for which every unit is obligated to make offers 
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



#split the data according the lookback and the lookforward that we want
# in this project our goal is to predict the offers for 7-day ahead so the lookforward must be 7*48=336 (48 half-hour offers per day) 
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


#calculate the accuracy score by counting the times the the predictions are correct
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




df, Date = read_names(energy_type)
entityname=df.names.values

#read the offers of every unit and save it in a dataframe
for i in entityname:
  vars()[i] = pd.read_excel('/content/drive/MyDrive/BE/BE_'+str(energy_type)+'_New_New/%s.xlsx' % i)
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

error_index = entityname.copy()
error_index = np.append(error_index, 'mean')

error_name = []
error_name.append('Accuracy')



g=0
tot_error = pd.DataFrame(index=np.arange(len(error_index)), columns=np.arange(1))
#tot_accuracy = pd.DataFrame(index=np.arange(len(error_index)), columns=np.arange(2))
tot_pred = 0
weeks_of_pred = 7
for i in entityname:
  entity = vars()[i]
  entity = entity.loc['2021-02-13':'2021-07-14']
  variable= entity.loc[:, 'VariableCost']
  availability = entity.loc[:, 'Availability']
  max_steps = max(entity.NumOfSteps)
  entity1 = entity.iloc[:, 3:3+max_steps]
  max_list = entity1.max(axis=0).values
  min_list = entity1.min(axis=0).values
  max_quantity = entity1.iloc[0:1, :].sum(axis=1)
  max_quantity = max_quantity.values
  #print(max_quantity)
  print(max_steps)

  if max_steps > 0:
    tot_pred +=1


    #LSTM
    k = classify_steps(max_steps, entity.NumOfSteps)
    k.index = date1.index
    dataset1 = k
    #dataset1 = pd.concat([k, date1, variable, availability], axis=1)
    cl = dataset1.values.reshape(dataset1.shape[0],len(dataset1.columns))
    cl
    #cl = scl.fit_transform(cl)
    lookback =7*48
    lookforward = 7*48
    X,y = split_sequences(cl, lookback, lookforward, max_steps)
    for k in range(1, X.shape[0], 7*48):
      if abs(X.shape[0]*0.8-k) < abs(X.shape[0]*0.8-(k-1)):
        split = k-1
    X_train,X_test = X[:split], X[split:X.shape[0]]
    y_train,y_test = y[:split],y[split:y.shape[0]]
    X_test_model = X_test[0:len(X_test)-336, :, :]
    X_test_pre = X_test[len(X_test)-336:, :, :]
    y_test_model = y_test[0:len(y_test)-336, :, :]
    y_test_pre = y_test[len(y_test)-336:, :, :]
    
    
    model = Sequential()
    model.add(LSTM(100, input_shape=(X_train.shape[1], 
                                    X_train.shape[2]),
                   return_sequences=True))
    model.add(Dropout(0.05))
    model.add(LSTM(80, return_sequences = True))
    model.add(Dropout(0.05))
    model.add(LSTM(80, return_sequences = True))
    model.add(Dense(units=48, activation='softmax'))
    model.add(Dense(units=max_steps, activation='softmax'))
    adam = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
    X_test_pre = X_test_pre.reshape(X_test_pre.shape[0], X_test_pre.shape[1], X_test_pre.shape[2])
    X_test_model = X_test_model.reshape(X_test_model.shape[0], X_test_model.shape[1], X_test_model.shape[2])
    #print(model.summary())
    history = model.fit(X_train,y_train,epochs=100, validation_data=(X_test_model, y_test_model), 
                        callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)], shuffle=False)
    
    Xt = model.predict(X_test_pre)
    
    m = Xt.max(axis=2)

    number = Xt[:,:, 0]
    number.shape
    m = Xt.max(axis=2)

    #by predicting half-hour offers for 7-days ahead a lot of prediction are doublicated so we reshaping the table with the data by keeping only the data we want
    pred = Xt[:,:, 0]
    for z in range(0, Xt.shape[0]):
      
      for k in range(0, Xt.shape[1]):
        
        for j in range(0,max_steps):
          if Xt[z,k,j] == m[z,k]:
            Xt[z,k,j] = 1
            pred[z,k] = j+1
          if Xt[z,k,j] !=1:
            Xt[z,k,j] = 0


    real = y_test_pre[:,:,0]
    for z in range(0, Xt.shape[0]):
      
      for k in range(0, Xt.shape[1]):
        
        for j in range(0,max_steps):
          if y_test_pre[z,k,j] == 1:
            real[z,k] = j+1


    table= []
    for z in range(0, pred.shape[0]):
      for j in range(0, pred.shape[1]):
        table.append(pred[z,j])
        table.append(real[z,j])



    final = np.array(table)
    final = final.reshape(int(len(table)/672), 672)
    final = pd.DataFrame(final)
    final.to_excel('/content/drive/MyDrive/BE/BE_'+str(energy_type)+'_New/Classification/pred_full/'+str(i)+'.xlsx')




    final_accuracy = pd.DataFrame(index=np.arange(weeks_of_pred+1), columns=np.arange(2))
    for t in range(0, 7):

      a = pd.DataFrame(index=np.arange(pred.shape[0]), columns=np.arange(2))
      
            
      row=0
            
          
      for j in range(0, pred.shape[1]):
        a.iat[row, 0]= pred[48*t+47,j]
        a.iat[row, 1] = real[48*t+47,j]
        row = row+1
      #a = a.drop(a.index[len(a)-1]) 
          #a.columns = name
      d = entity.iloc[(len(entity)-336-(6-t)*48):(len(entity)-(6-t)*48), :]
      
      a.index = d.index
      a.columns = ['Predicted', 'Real']
     
      
      a.to_excel('/content/drive/MyDrive/BE/BE_'+str(energy_type)+'_New/Classification/predictions/'+str(i)+'at'+str(t)+'day.xlsx')
      

      col = 1    
      index_list = a.index.tolist()
      first_day = index_list[0].day
      first_day_month = index_list[0].month
      last_day = index_list[len(index_list)-1].day
      last_day_month = index_list[len(index_list)-1].month
      date = '{}/{}-{}/{}'.format(first_day, first_day_month, last_day, last_day_month)
      
      
      #mae.iat[t,0] = date
      #mae.iat[t,1] = accuracy_score_manual(a.iloc[:, 0], a.iloc[:, 1])
      final_accuracy.iat[t,0] = date
      final_accuracy.iat[t, 1] = accuracy_score_manual(a.iloc[:, 0], a.iloc[:, 1])
        
    mae_col_name = []
    mae_col_name.append('Date')

    mae_col_name.append('Accuracy')
    final_accuracy.columns = mae_col_name
    final_accuracy =  final_accuracy.set_index('Date')
    final_accuracy_sum = final_accuracy.sum().tolist()
    final_accuracy.iat[weeks_of_pred, 0] = 'mean error'

    tot_error.iat[g, 0] =final_accuracy_sum[len(final_accuracy.columns)-1]/(weeks_of_pred)
    final_accuracy.iat[weeks_of_pred, len(final_accuracy.columns)-1] = final_accuracy_sum[len(final_accuracy.columns)-1]/(weeks_of_pred)
    final_accuracy.to_excel('/content/drive/MyDrive/BE/BE_'+str(energy_type)+'_New/Classification/error/'+str(i)+'.xlsx')
    print(g, i)
    g+=1


#calculate the total mean error of all unit's offers prediction
tot_mae_sum = tot_error.sum().tolist()
for k in range(len(tot_error.columns)):
  tot_error.iat[len(tot_error)-1,k] = tot_mae_sum[k]/tot_pred


tot_error.columns = error_name
tot_error.index = error_index
tot_error.to_excel('/content/drive/MyDrive/BE/BE_'+str(energy_type)+'_New/Classification/final 1layers 10units only offers.xlsx')