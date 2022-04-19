# Prediction of bids for backup energy in the Greek Balancing Market using machine learning techniques


Description 

The aim of this project is the short-term forecast for 7 days of bids for upward and downward balancing energy which will be made by the participants in the Hellenic electricity Balancing Market. Due to the competitive nature of the market such an accurate forecast would have a variety of benefits for all participants. 
The approach to this prediction was carried out in two ways. First naive prediction models were used and then complex models using neural networks and decision trees. 	

Specifically, three categories of neural models were examined: Recurrent Neural Networks that use Long Short Term Memory neurons (LSTM), Autoencoder Recurrent Neural Networks that use Long Short Term Memory neurons (Autoencoder LSTM) and Artificial Neural Networks (ANN). The type of Random Forest Decision Trees was also examined. Using the above, four different models were created. Two Recurrent Neural Networks with Long Short Term Memory neurons, one of which with Encoder-Decoder and both performed prediction with regression techniques. Also an Artificial Neural Network and a Decision Tree, both of which use as input the features exported by a Recurrent Neural Network with Long Short Term Memory neurons that performs prediction using classification techniques. After that, the performance of the models is studied, the errors resulting from the different forecasting models are compared and the optimal prediction method of the offers of each producer is selected.
