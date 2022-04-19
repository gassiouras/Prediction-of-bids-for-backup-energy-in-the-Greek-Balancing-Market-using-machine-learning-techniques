# Prediction of bids for backup energy in the Greek Balancing Market using machine learning techniques


## Description 

The aim of this project is the short-term forecast for 7 days of bids for upward and downward balancing energy which will be made by the participants in the Hellenic electricity Balancing Market. Due to the competitive nature of the market such an accurate forecast would have a variety of benefits for all participants. 
The approach to this prediction was carried out in two ways. First naive prediction models were used and then complex models using neural networks and decision trees. 	

Specifically, three categories of neural models were examined: Recurrent Neural Networks that use Long Short Term Memory neurons (LSTM), Autoencoder Recurrent Neural Networks that use Long Short Term Memory neurons (Autoencoder LSTM) and Artificial Neural Networks (ANN). The type of Random Forest Decision Trees was also examined. Using the above, four different models were created. Two Recurrent Neural Networks with Long Short Term Memory neurons, one of which with Encoder-Decoder and both performed prediction with regression techniques. Also an Artificial Neural Network and a Decision Tree, both of which use as input the features exported by a Recurrent Neural Network with Long Short Term Memory neurons that performs prediction using classification techniques. After that, the performance of the models is studied, the errors resulting from the different forecasting models are compared and the optimal prediction method of the offers of each producer is selected.

### 1. Data and the cleaning programs:

[Data Processing](https://github.com/gassiouras/Prediction-of-bids-for-backup-energy-in-the-Greek-Balancing-Market-using-machine-learning-techniques/blob/65e5c13c15066d45132f7761c83cd612a981abcb/Data%20Procesing.py)

With this script we matching the offers that every unit had made and create separate files for every unit in order to train later our model and make predictions.

_Tip_

I recommend that you have all the files on the drive and link the python script with it so that you do not take up a lot of PC memory.

### 2. Predictive models



There are three different ways that I used to predict the offers.

* Three naive models: The first one copy the last day offers that each unit made to predict the next week. The secont one copy the last week offers to predict the next. The third one set the predicted offers equal with the mean price of the last weeks offers
* LSTM-regression and Encoder-Decoder LSTM models: Both of them are trained with the dataset and makes prediction for both the quantity and the price of each offer. The predicted values for the quantity, given by each model, are classified using some rules that the units are obligated to follow.
* LSTM-ANN and LSTM-Random Forest Trees models: In this method I use the LSTM model to predict the number of the step that the unit will use for its offer. Then based on the number of steps the ANN and the Random Forest Trees models predict the actuall Quantity and Price of every offer.




 
