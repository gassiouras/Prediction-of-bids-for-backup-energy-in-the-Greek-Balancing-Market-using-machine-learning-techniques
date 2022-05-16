import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
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



def quantity_values(entity):
  quantity = entity.loc[:, 'Step1Quantity':'Step10Quantity']
  column_values = quantity['Step1Quantity'].values
  Step1Val =  np.unique(column_values)
  column_values = quantity['Step2Quantity'].values
  Step2Val =  np.unique(column_values)
  column_values = quantity['Step3Quantity'].values
  Step3Val =  np.unique(column_values)
  column_values = quantity['Step4Quantity'].values
  Step4Val =  np.unique(column_values)
  column_values = quantity['Step5Quantity'].values
  Step5Val =  np.unique(column_values)
  column_values = quantity['Step6Quantity'].values
  Step6Val =  np.unique(column_values)
  column_values = quantity['Step7Quantity'].values
  Step7Val =  np.unique(column_values)
  column_values = quantity['Step8Quantity'].values
  Step8Val =  np.unique(column_values)
  column_values = quantity['Step9Quantity'].values
  Step9Val =  np.unique(column_values)
  column_values = quantity['Step10Quantity'].values
  Step10Val =  np.unique(column_values)
  val1 = pd.DataFrame(Step1Val)
  val2 = pd.DataFrame(Step2Val)
  val3 = pd.DataFrame(Step3Val)
  val4 = pd.DataFrame(Step4Val)
  val5 = pd.DataFrame(Step5Val)
  val6 = pd.DataFrame(Step6Val)
  val7 = pd.DataFrame(Step7Val)
  val8 = pd.DataFrame(Step8Val)
  val9 = pd.DataFrame(Step9Val)
  val10 = pd.DataFrame(Step10Val)
  val = pd.concat([val1, val2, val3, val4, val5, val6, val7, val8, val9, val10], axis=1)
  val= val.fillna(0)
  return val


def mean_absolute_percentage_error_custom(y_true, y_pred):
  return 100*np.mean(np.abs(percentage_error(np.asarray(y_true), np.asarray(y_pred))))  


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



def accuracy_score_manual(pred, real):
  k=0
  for i in range(0, len(pred)):
    if pred.iat[i] == real.iat[i]:
      k+=1
  return k*100/len(pred)


#ask user to select the type of energy and the type of variable he want to predict
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



error_index = entityname.copy()
error_index = np.append(error_index, 'mean')



error_name = []
for k in range(1,11):
  error_name.append('MAE')
  error_name.append('RMSE')
error_name.append('Accuracy')


g=0
tot_error = pd.DataFrame(index=np.arange(len(error_index)), columns=np.arange(21))
tot_pred = 0
weeks_of_pred = 7

if var_type==0:
  col_num = 0 #col num=0 for quantity and clo_num=10 for prices
  diagram_names = 'Quantity'
  save_names = 'Quantity'
else:
  col_num = 10 #col num=0 for quantity and clo_num=10 for prices
  diagram_names = 'Price'
  save_names = 'Price'

for i in entityname:
  entity = vars()[i].loc[:, 'Step1Quantity':'Step10Price']
  entity1 = entity.loc['2021-02-13':'2021-07-14']

  #number of steps 
  values = quantity_values(entity)
  for j in range(0, len(values.columns)):
    if values.iloc[:, j].sum()==0:
      final_step=j
      break
    else:
      final_step=10
  #mae index name
  name=[]
  for j in range(0, final_step):
    name.append('Step'+str(j+1)+'Predicted')
    name.append('Step'+str(j+1)+'Real')
  name.append('StepNumPred')
  name.append('StepNumReal')
  mae = pd.DataFrame(index=np.arange(weeks_of_pred+1), columns=np.arange(2*final_step+2))
  mae_col_name = []
  mae_col_name.append('Date')
  for j in range(1, final_step+1):
    mae_col_name.append('MAE')
    mae_col_name.append('RMSE')
  mae_col_name.append('Accuracy')
  mae.columns = mae_col_name



  for t in range(6, -1, -1):
      week = entity1.iloc[len(entity1)-48*(7+t):len(entity1)-48*t, col_num:col_num+final_step]
      week0 = entity1.iloc[len(entity1)-48*(8+t):len(entity1)-48*(t+7), col_num:col_num+final_step]
      
      naive = pd.DataFrame(index=np.arange(len(week)), columns=np.arange(final_step*2+2))

      for j in range(0, final_step):
        cor = 0
        p = 0
        for k in range(0, len(week)):
          

          naive.iat[k,j*2] = week0.iat[p,j]
          naive.iat[k, j*2+1] = week.iat[k, j]
          if p<47:
            p+=1
          else:
            p=0
          if naive.iat[k,j*2] == naive.iat[k, j*2+1]:
            cor = cor+1

      for j in range(len(naive)):
        pr = 0
        rl =0
        for k in range(int(len(naive.columns)/2)-1):
          if naive.iat[j,2*k]>0:
            pr+=1
          if naive.iat[j,2*k+1]>0:
            rl+=1
        naive.iat[j, 2*final_step] = pr
        naive.iat[j,2*final_step+1] = rl

      naive.index = week.index
      naive.columns = name
      naive.to_excel('/content/drive/MyDrive/BE/BE_'+str(energy_type)+'_New/naive/lastday/'+str(save_names)+'/naive_predictions/'+str(i)+' prediction at '+str(6-t)+'day.xlsx')

      col = 1
      index_list = week.index.tolist()
      first_day = index_list[0].day
      first_day_month = index_list[0].month
      last_day = index_list[len(index_list)-1].day
      last_day_month = index_list[len(index_list)-1].month
      date = '{}/{}-{}/{}'.format(first_day, first_day_month, last_day, last_day_month)
      mae.iat[t,0] = date
      #print(naive.iloc[:, 2])
      for k in range(0, final_step):
        mae.iat[6-t, 2*k+1] = mean_absolute_error(naive.iloc[:, 2*k+1], naive.iloc[:, 2*k])
        mae.iat[6-t, 2*k+2] = sqrt(mean_squared_error(naive.iloc[:, 2*k+1], naive.iloc[:, 2*k]))
      mae.iat[6-t, 2*final_step+1] = accuracy_score_manual(naive.iloc[:,2*final_step], naive.iloc[:, 2*final_step+1])
  mae.iat[weeks_of_pred,0] = 'mean error'
  mae  = mae.set_index('Date')
  mae_sum = mae.sum().tolist()
  for k in range(0, len(mae.columns)):
    mae.iat[weeks_of_pred, k] = mae_sum[k]/(weeks_of_pred)
    if k < len(mae.columns)-1:
      tot_error.iat[g, k] = mae_sum[k]/(weeks_of_pred)
    else:
      tot_error.iat[g, 20] = mae_sum[k]/(weeks_of_pred)  
  mae.to_excel('/content/drive/MyDrive/BE/BE_'+str(energy_type)+'_New/naive/lastday/'+str(save_names)+'/naive_error/'+str(i)+'.xlsx')
  print(i)
  g+=1
  tot_pred +=1