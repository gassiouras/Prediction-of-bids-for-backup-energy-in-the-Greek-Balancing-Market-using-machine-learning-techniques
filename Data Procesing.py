from tkinter import Variable
from tkinter.font import names
from django import dispatch
import pandas as pd
import numpy as np
import datetime
import math


class data_preprocess:
    #load data of BE_offers, availability and variable cost
    def data_load (self):
        offers_old = pd.read_csv(r"C:\Users\Γιάννης\Downloads\BE Matched Offers Quantity Differences (2021-01-01 to 2021-07-14) (2).csv", low_memory=False)
        availability_old = pd.read_excel(r"C:\Users\Γιάννης\Downloads\Variable.xlsx")
        variableCost_old = pd.read_csv(r"C:\Users\Γιάννης\Downloads\VariableCosts[2021-01-01 to 2021-07-14].csv", low_memory=False)
        return (offers_old, availability_old, variableCost_old)
    


    #create a excel file with the names of the units which partisipated in the bidings from 1/1/2021 to 14/7/2021
    def units_names(self):
        offers, a, var = self.data_load()
        names = offers['Name'].unique()
        names = pd.DataFrame(names)
        names.to_excel(r"C:\Users\Γιάννης\Downloads\Names.xlsx")
        return (names)



    #change the type of DispatchDay columns to datetime In order to process the files easier
    def fix_datetime(self):
        offers, availability, variableCost = self.data_load()


        offers['DispatchDay'] = pd.to_datetime(offers['DispatchDay'])
        offers['DispatchDay'] = pd.to_datetime(offers['DispatchDay'], format='%Y - %m - %d')
        offers = offers.sort_values(by=['DispatchDay', 'DispatchPeriod'], ascending=True)


        availability['DispatchDay'] = pd.to_datetime(availability['DispatchDay'])
        availability['DispatchDay'] = pd.to_datetime(availability['DispatchDay'], format='%Y - %m - %d')
        availability = availability.sort_values(by=['DispatchDay', 'DispatchPeriod'], ascending=True)
        
        variableCost['DispatchDay'] = pd.to_datetime(variableCost['DispatchDay'])
        variableCost['DispatchDay'] = pd.to_datetime(variableCost['DispatchDay'], format='%Y - %m - %d')
        variableCost = variableCost.sort_values(by=['DispatchDay', 'DispatchPeriod'], ascending=True)


        return (offers, availability, variableCost)

    #in this function we get the position of every argument we want in the csv files in order to use 
    def get_col_val(self, energy_type):
        self.energy_type = energy_type


        offers, availability, variableCost = self.fix_datetime()
        col_list=offers.columns.to_list()



        name = col_list.index('Name')
        date = col_list.index('DispatchDay')
        period = col_list.index('DispatchPeriod')
        energy_type = col_list.index(energy_type)
        
        price = col_list.index('Price')
        quantity = col_list.index('Quantity')
        step = col_list.index('Step')

        av_list = availability.columns.to_list()
        av_name = av_list.index('Name')
        av_date = av_list.index('DispatchDay')
        av_period = av_list.index('DispatchPeriod')
        av = av_list.index('Value')

        va_list = variableCost.columns.to_list()
        va_name = va_list.index('UnitName')
        va_date = va_list.index('DispatchDay')
        va_period = va_list.index('DispatchPeriod')
        va = va_list.index('FinalVariableCost')


        return(name, date, period, energy_type, price, quantity, step, av_name, av_date, av_period, av, va_name, va_date, va_period, va)
    def create_frame(self):
        offers, availability, variableCost = self.fix_datetime()
        #append the unit AG_DIMITRIOS1 from whitch the data are wrong
        entityname = offers['Name'].value_counts().index.tolist()
        entityname.append('AG_DIMITRIOS1')


        #create a DataFrame with 24 columns and as many rows as the offers that each unit has made
        dispatchday = offers['Dispatchday'].value_counts().index.tolist()
        dispatchday.sort()

        Day=[]
        Period=[]

        for i in dispatchday:
            for j in range(1, 49):
                Day.append(i)
                Period.append(j)

        d = offers.DataFrame({'DispatchDay': Day[:], 'DispatchPeriod': Period[:]})
        
        a = offers.DataFrame(index=np.arange(len(d)), columns=np.arange(24))
        a = a.drop([0], axis=1)

        new_frame = pd.DataFrame(None)
        new_frame = pd.concat([d, a], axis=1)
        new_frame = new_frame.set_axis(['DispatchDay', 'DispatchPeriod', 'Step1Quantity', 'Step2Quantity', 'Step3Quantity','Step4Quantity',
                            'Step5Quantity','Step6Quantity','Step7Quantity','Step8Quantity','Step9Quantity',
                            'Step10Quantity','Step1Price', 'Step2Price', 'Step3Price','Step4Price',
                            'Step5Price','Step6Price','Step7Price','Step8Price','Step9Price',
                            'Step10Price', 'Date', 'Availability', 'VariableCost'], axis=1, inplace=False)
        new_frame['Date'] = pd.to_datetime(new_frame['DispatchDay'], format='%Y - %m - %d - %h - %min')


        return entityname, new_frame
    
# this class creates excel files with the offers of every unit separate
# if we want the offers for Up reserve energy we set the energy_type="Up" and for Down reserve energy offers energy_type='Down'
#if we want to create the excel file for only a specific unit we set unit='The name of the unit we want'
class data_process(data_preprocess):
    def files_creator(self, energy_type, unit='All'):
        self.unit = unit
        #get the fixed data
        df, availability, variableCost = data_preprocess().fix_datetime()
        entityname, newframe = data_preprocess().create_frame()

        #check if we want a specific unit else entityname contains every unit name
        if unit != 'All':
            entityname = unit
        

        self.energy_type = energy_type
        
        #call get_col_val funtion to get the position of every argument we want  
        name, date, period, energy_type, price, quantity, step, av_name, av_date, av_period, av, va_name, va_date, va_period, va = data_preprocess().get_col_val(energy_type)
        
        if energy_type=='Up':
            energy_type1=True
        else:
            energy_type1=False

        list = (5,6,7,8)
        g = len(entityname)
        #for every unit we search the originals files get the offers of the unit and the type of the offer (which step is it and for which day and period)
        for i in entityname:
            vars()[i] = newframe
        
            for j in range(0, len(df)):
                if df.iat[j, name]==i and df.iat[j, energy_type]==energy_type1:
                    if df.iat[j,step]==1:
                        vars()[i].loc[(vars()[i]['DispatchDay']==df.iat[j,date]) & (vars()[i]['DispatchPeriod']==int(df.iat[j,period])), ['Step1Quantity', 'Step1Price']  ] = df.iat[j,quantity], df.iat[j,price]
                    elif df.iat[j,step]==2:
                        vars()[i].loc[(vars()[i]['DispatchDay']==df.iat[j,date]) & (vars()[i]['DispatchPeriod']==int(df.iat[j,period])), ['Step2Quantity', 'Step2Price']  ] = df.iat[j,quantity], df.iat[j,price]

                    elif df.iat[j,step]==3:
                        vars()[i].loc[(vars()[i]['DispatchDay']==df.iat[j,date]) & (vars()[i]['DispatchPeriod']==int(df.iat[j,period])), ['Step3Quantity', 'Step3Price']  ] = df.iat[j,quantity], df.iat[j,price]

                    elif df.iat[j,step]==4:
                        vars()[i].loc[(vars()[i]['DispatchDay']==df.iat[j,date]) & (vars()[i]['DispatchPeriod']==int(df.iat[j,period])), ['Step4Quantity', 'Step4Price']  ] = df.iat[j,quantity], df.iat[j,price]

                    elif df.iat[j,step]==5:
                        vars()[i].loc[(vars()[i]['DispatchDay']==df.iat[j,date]) & (vars()[i]['DispatchPeriod']==int(df.iat[j,period])), ['Step5Quantity', 'Step5Price']  ] = df.iat[j,quantity], df.iat[j,price]

                    elif df.iat[j,step]==6:
                        vars()[i].loc[(vars()[i]['DispatchDay']==df.iat[j,date]) & (vars()[i]['DispatchPeriod']==int(df.iat[j,period])), ['Step6Quantity', 'Step6Price']  ] = df.iat[j,quantity], df.iat[j,price]

                    elif df.iat[j,step]==7:
                        vars()[i].loc[(vars()[i]['DispatchDay']==df.iat[j,date]) & (vars()[i]['DispatchPeriod']==int(df.iat[j,period])), ['Step7Quantity', 'Step7Price']  ] = df.iat[j,quantity], df.iat[j,price]

                    elif df.iat[j,step]==8:
                        vars()[i].loc[(vars()[i]['DispatchDay']==df.iat[j,date]) & (vars()[i]['DispatchPeriod']==int(df.iat[j,period])), ['Step8Quantity', 'Step8Price']  ] = df.iat[j,quantity], df.iat[j,price]

                    elif df.iat[j,step]==9:
                        vars()[i].loc[(vars()[i]['DispatchDay']==df.iat[j,date]) & (vars()[i]['DispatchPeriod']==int(df.iat[j,period])), ['Step9Quantity', 'Step9Price']  ] = df.iat[j,quantity], df.iat[j,price]

                    elif df.iat[j,step]==10:
                        vars()[i].loc[(vars()[i]['DispatchDay']==df.iat[j,date]) & (vars()[i]['DispatchPeriod']==int(df.iat[j,period])), ['Step10Quantity', 'Step10Price']  ] = df.iat[j,quantity], df.iat[j,price]
            
            #we match the availability and the variable cost of the unit for every period and whrite them in the last tow columns of the file
            for k in range(0, len(availability)):
                if availability.iat[k, av_name]==i:
                    vars()[i].loc[(vars()[i]['DispatchDay']==availability.iat[k,av_date]) & (vars()[i]['DispatchPeriod']==int(availability.iat[k,av_period])), 'Availability'  ] = availability.iat[k,av]
            for k in range(0, len(variableCost)):
                if variableCost.iat[k, va_name]==i:
                    vars()[i].loc[(vars()[i]['DispatchDay']==variableCost.iat[k,va_date]) & (vars()[i]['DispatchPeriod']==int(variableCost.iat[k,va_period])), 'VariableCost'  ] = variableCost.iat[k,va]


            # create a colum with the date and time like 01/01/2021 00:00:00 by match DispatchDay and DispatchPeriod
            for k in range(0, len(vars()[i])):
                if vars()[i].iat[k,1]==0 or vars()[i].iat[k,1]==1:
                    continue
                elif vars()[i].iat[k,1]%2==0:
                    h = int((vars()[i].iat[k,1]-2)/2)
                    vars()[i].iat[k,22] = pd.Timestamp(vars()[i].iat[k,22]).replace(hour=h, minute=30)
                elif vars()[i].iat[k,1]%2!=0:
                    h = int((vars()[i].iat[k,1]-1)/2)
                    vars()[i].iat[k,22] = pd.Timestamp(vars()[i].iat[k,22]).replace(hour=h) 
                #fix the 28/03 problem when the hour changed and we had a gap in the data
                if vars()[i].iat[k,0] == pd.Timestamp(2021, 3, 28):
                    if vars()[i].iat[k,1] in list :
                        for j in range(2,22):
                            vars()[i].iat[k,j] = vars()[i].iat[k-1,j]  


            #count the number of the step that the unit use and save the number in the last column                     
            vars()[i]['NumOfSteps'] = vars()[i]['Step1Quantity'].copy()
            for j in range(0, len(vars()[i])):
                t=0
                for k in range(3, 13):
                    if vars()[i].iat[j,k] != 0:
                        t +=1
    
            vars()[i].iat[j, len(vars()[i].columns)-1] = t   



            g -= 1
            #create the excel file
            vars()[i].to_excel(r'C:\Users\Γιάννης\Downloads/BE_'+str(energy_type)+'_New/' +str(i)+ '.xlsx')
            print(g, i)
            return g


print(data_process().files_creator('Up', 'PROTERGIA_CC'))
