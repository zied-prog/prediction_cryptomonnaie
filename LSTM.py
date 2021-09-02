#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install yfinance
import yfinance as yf
import datetime

import pandas
import math

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import pandas_datareader as web


# In[2]:


## Lire les prix de l'Ethereum
start = datetime.datetime(2019, 1, 1)
end = datetime.datetime(2021, 7, 31)
hist_data = web.DataReader("ETH-USD", 'yahoo', start, end)
hist_data


# In[3]:


#Creation d'une nouvelle data frame
hist_data=hist_data.filter(['Adj Close'])
hist_data.tail()


# In[4]:


#Convertion de la dataframe en numpy array
Dataset = hist_data.values


# In[5]:


#Avoir le nombre de lignes pour entrainer le model
training_data_len = math.ceil(len(hist_data)*.8)
training_data_len


# In[6]:


# Normaliser la data
scaler = MinMaxScaler()
Scaled_data = scaler.fit_transform(hist_data)

Scaled_data,Scaled_data.shape


# In[7]:


# Creation du jeu d'entrainement

Train_data = Scaled_data[0:training_data_len,:]

# Separation de la data en jeu d'entrainement 
x_train = []
y_train = []

for i in range(60, len(Train_data)):
  x_train.append(Train_data[i-60:i,0])
  y_train.append(Train_data[i,0])
  if i<=61:
    print(x_train)
    print(y_train)
    print()
print(len(x_train))
print(len(y_train))


# In[8]:


#Convertion de x_train et y_train en numpy array
x_train, y_train = np.array(x_train), np.array(y_train)


# In[9]:


#Redimentionner la data en 3 dimentions
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
x_train.shape


# In[10]:


# Construction du model LSTM
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))


# In[11]:


#Compiler le model
model.compile(optimizer='adam', loss='mean_squared_error')


# In[13]:


# Entrainer le model
model.fit(x_train, y_train, batch_size=1, epochs=10)


# In[14]:


## Creation du jeu de donnée de test

test_data = Scaled_data[training_data_len-60: , :]
#Creation de jeu de donnée x_test and y_test
x_test = []
y_test = Dataset[training_data_len:, : ]
for i in range(60, len(test_data)):
  x_test.append(test_data[i-60:i, 0])


# In[15]:


#Convertion de x_test en numpy array
x_test = np.array(x_test)


# In[16]:


#Redimentionner la data en 3 dimentions
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))


# In[17]:


### Faisons quelques prédictions
train_predict=model.predict(x_train)
test_predict=model.predict(x_test)


# In[18]:


##Transformback to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)


# In[19]:


# Avoir les predictions
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)


# In[20]:


print(predictions[0:10], y_test[0:10])


# In[21]:


# Avoir le MSE & RMSE
mse = np.mean(predictions-y_test)**2
rmse = np.sqrt(np.mean(predictions-y_test)**2)
mse, rmse


# In[22]:


mape = np.mean(np.abs(predictions- y_test)/np.abs(y_test))  
mape


# In[23]:


mae = np.mean(np.abs(predictions- y_test))  
mae


# In[27]:


mpe = np.mean((predictions - y_test)/y_test)  
mpe


# In[24]:


# Plot la data
train = hist_data[:training_data_len]
valid = hist_data[training_data_len:]
valid['Predictions'] = predictions

# Visualiser la data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price ', fontsize=18)
plt.plot(train['Adj Close'])
plt.plot(valid[['Adj Close','Predictions']])
plt.legend(['True', 'Val', 'Predictions'], loc = 'lower right')
plt.show()


# In[25]:


# Avoir les cotations
apple_quote = web.DataReader('ETH-USD', data_source='yahoo', start='2019-01-01', end='2021-7-31')
#Creation d'une nouvelle data frame
new_df = apple_quote.filter(['Adj Close'])
# Prendre les dernier 60j Adj Close et convertir la nouvelle data frame en tableau
last_60_days = new_df[-60:].values
#Normaliser la data
last_60_days_scaled = scaler.transform(last_60_days)
#Creation d'une liste vide
X_test = []
#Ajouter les derniers 60 j 
X_test.append(last_60_days_scaled)
#Convertir X_test data set en un numpy tableau
X_test = np.array(X_test)
#Redimensionner la data
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
# Prendre le prix normalisé
pred_price = model.predict(X_test)
#Dénormaliser le prix
pred_price = scaler.inverse_transform(pred_price)
print(pred_price)


# In[26]:


# La cotation du J+1
ETH_quote2 = web.DataReader('ETH-USD', data_source='yahoo', start='2021-8-2', end='2021-8-2')
ETH_quote2


# In[ ]:




