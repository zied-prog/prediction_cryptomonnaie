#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pandas_datareader.data as pdr
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import warnings
import itertools
import statsmodels.api as sm
from pylab import rcParams
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
import datetime 
from datetime import timedelta
#!pip install yfinance
import yfinance as yf


# In[11]:


start = datetime.datetime(2019,1,1)
end = datetime.datetime(2021,7,31)

ETH = yf.download('ETH-USD', start, end)


# In[12]:


ETH


# In[13]:


ETH_DATA=ETH.filter(['Adj Close'])
ETH_DATA.tail()


# In[14]:


ETH.dtypes 
ETH.info()
ETH.describe()


# In[16]:


Data_viz = ETH.plot(y= 'Adj Close', figsize=(12,6), legend=True, grid=True, use_index=True)
plt.show()


# In[17]:


ETH_price = ETH['Adj Close']


# In[18]:


# Tracer la moyenne mobile et l’écart-type mobile
ETH_price.rolling(12).mean().plot(figsize=(20,10), linewidth=5, fontsize=20)
plt.xlabel('time', fontsize=20);

rolling_mean = ETH_price.rolling(window = 12).mean()
rolling_std = ETH_price.rolling(window = 12).std()

plt.plot(ETH_price, color = 'blue', label = 'Origine')
plt.plot(rolling_mean, color = 'red', label = 'Moyenne mobile')
plt.plot(rolling_std, color = 'green', label = 'Ecart-type mobile')
plt.legend(loc = 'best')
plt.title('Moyenne et Ecart-type mobiles')
plt.show()


# In[19]:


# Test de Dickey-Fuller augmenté qui est un test de stationnarité:
Test_ADF = adfuller(ETH_price)

print('Statistiques ADF : {}'.format(Test_ADF[0]))
print('p-value : {}'.format(Test_ADF[1]))


# In[20]:


ETH_price_1=  ETH_price-ETH_price.shift()
ETH_price_1.dropna(inplace=True)  
ETH_price_1.head()


# In[21]:


ETH_price_1.rolling(12).mean().plot(figsize=(20,10), linewidth=5, fontsize=20)
plt.xlabel('time', fontsize=20);

rolling_mean = ETH_price_1.rolling(window = 12).mean()
rolling_std = ETH_price_1.rolling(window = 12).std()

plt.plot(ETH_price_1, color = 'blue', label = 'Origine')
plt.plot(rolling_mean, color = 'red', label = 'Moyenne mobile')
plt.plot(rolling_std, color = 'green', label = 'Ecart-type mobile')
plt.legend(loc = 'best')
plt.title('Moyenne et Ecart-type mobiles de ETH data')
plt.show()


# In[22]:


Test_ADF = adfuller(ETH_price_1)

print('Statistiques ADF : {}'.format(Test_ADF[0]))
print('p-value : {}'.format(Test_ADF[1]))


# In[23]:


#  Convertion en moyennes hebdomadaires de nos séries chronologiques
ETH_price_2 = ETH_price_1.resample('W').mean()
ETH_price_2.head()


# In[24]:


# Somme des données manquantes
ETH_price_2.isnull().sum()


# In[25]:


ETH_price_2.plot(figsize=(15, 6))
plt.show()


# In[26]:


# Decomposition de la serie chronologique pour identifier la saisonnalité
rcParams['figure.figsize'] = 11, 9

decomposition = sm.tsa.seasonal_decompose(ETH_price_2, model='additive')
decomposition_plot = decomposition.plot()
plt.show()


# In[27]:


# Generation des différentes combinaisons de paramètres (p, d et q) qui peuvent prendre n'importe quelles valeurs entre 0 et 2 
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
parametre_pdq = [(x[0], x[1], x[2], 52) for x in list(itertools.product(p, d, q))]
parametre_pdq


# In[28]:


# Utilisation du modele SARIMA suite à l'identification de la saisonnalité

for parametre in pdq:
    for parametre_seasonal in parametre_pdq:
        try:
            model_SARIMA = sm.tsa.statespace.SARIMAX(ETH_price_2, order=parametre, seasonal_order=parametre_seasonal, enforce_stationarity=False,  enforce_invertibility=False)

            resultats = model_SARIMA.fit()

            print('ARIMA{}x{}52 - AIC:{}'.format(parametre, parametre_seasonal, resultats.aic))
        except:
            continue


# In[29]:


#La sortie du modèle est ARIMA(1, 1, 1)x(1, 1, 0, 52)52 et  la valeur AIC la plus faible est 330.11656280001256
model = sm.tsa.statespace.SARIMAX(ETH_price_2,order=(1, 1, 1),seasonal_order=(1, 1, 0, 12),enforce_stationarity=False, enforce_invertibility=False)
arima = model.fit()
print(arima.summary())


# In[30]:


arima.plot_diagnostics(figsize=(15, 12))
plt.show()


# In[31]:


# Construction d'un intervalle de confiance pour les paramètres ajustés.
prediction = arima.get_prediction(start=pd.to_datetime('2021-01-03'), dynamic=False)
prediction_intervalle = prediction.conf_int()


# In[ ]:





# In[32]:


pred = ETH_price_2['2019':].plot(label='observed')
prediction.predicted_mean.plot(ax=pred, label='Prediction', alpha=.7)

pred.fill_between(prediction_intervalle.index, prediction_intervalle.iloc[:, 0], prediction_intervalle.iloc[:, 1], color='k', alpha=.2)

pred.set_xlabel('Date')
pred.set_ylabel('Price')
plt.legend()

plt.show()


# In[33]:


forecast = arima.get_forecast(steps=8)
forecast_intervalle = forecast.conf_int()


# In[34]:


print(forecast_intervalle)


# In[35]:


ax = ETH_price_2.plot(label='observed', figsize=(20, 15))
forecast.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(forecast_intervalle.index, forecast_intervalle.iloc[:, 0], forecast_intervalle.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Price')

plt.legend()
plt.show()


# In[41]:


# Validation croisée hors du temps


# In[42]:


ETH_price_1 = ETH['Close']
ETH_price_2 = ETH_price_1-ETH_price_1.shift()
ETH_price_2.dropna(inplace=True)  

#split data 
train, test=ETH_price_2[:-20], ETH_price_2[-20:]


#Train 
train = train.resample('w').mean()
#Test 
test = test.resample('w').mean()


# In[43]:


test.describe()




# In[44]:


test


# In[45]:


p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
parametre_pdq = [(x[0], x[1], x[2], 52) for x in list(itertools.product(p, d, q))]


# In[46]:


for parametre in pdq:
    for parametre_seasonal in parametre_pdq:
        try:
            model_SARIMA = sm.tsa.statespace.SARIMAX(train, order=parametre, seasonal_order=parametre_seasonal, enforce_stationarity=False,  enforce_invertibility=False)

            resultats = model_SARIMA.fit()

            print('ARIMA{}x{}52 - AIC:{}'.format(parametre, parametre_seasonal, resultats.aic))
        except:
            continue


# In[47]:


model = sm.tsa.statespace.SARIMAX(train,order=(1, 1, 1),seasonal_order=(1, 1, 0, 52),enforce_stationarity=False, enforce_invertibility=False)
arima = model.fit()
print(arima.summary())


# In[49]:


arima.plot_diagnostics(figsize=(15, 12))
plt.show()


# In[50]:


prediction = arima.get_prediction(start=pd.to_datetime('2021-01-03'), dynamic=False)
prediction_intervalle = prediction.conf_int() 


# In[51]:


pred = train['2019':].plot(label='observed')
prediction.predicted_mean.plot(ax=pred, label='Prediction', alpha=.7)

pred.fill_between(prediction_intervalle.index, prediction_intervalle.iloc[:, 0], prediction_intervalle.iloc[:, 1], color='k', alpha=.2)

pred.set_xlabel('Date')
pred.set_ylabel('Price')
plt.legend()

plt.show()


# In[52]:


forecast = arima.get_forecast(steps=4)
forecast_intervalle = forecast.conf_int()

ax = train.plot(label='observed', figsize=(20, 15))
forecast.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(forecast_intervalle.index, forecast_intervalle.iloc[:, 0], forecast_intervalle.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Price')

plt.legend()
plt.show()


# In[58]:


Prediction =(prediction_intervalle["lower Close"]+prediction_intervalle["upper Close"]) /2
Prediction.index


# In[59]:


#RMSE
rmse = np.mean((Prediction - test)**2)**.5
rmse


# In[60]:


#MAPE
mape = np.mean(np.abs(Prediction- test)/np.abs(test))  
mape


# In[61]:


#MAE
mae = np.mean(np.abs(Prediction - test))  
mae


# In[62]:


#MPE 
mpe = np.mean((Prediction - test)/test)  
mpe


# In[ ]:




