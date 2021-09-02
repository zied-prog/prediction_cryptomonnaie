#!/usr/bin/env python
# coding: utf-8

# In[28]:


pip install pandas-datareader


# In[210]:


pip install kmodes


# In[301]:


#import des librairies 
import os
import sys
import requests

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import pandas_datareader.data as pdr

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import accuracy_score, classification_report
import talib as ta


# In[314]:


# affectation de la date d'aujourd'hui comme date de fin
end_date = datetime.today()


# In[315]:


# affectation de la date dé début
start_date = datetime(end_date.year,end_date.month-1,end_date.day)


# In[316]:


#importation des données historiques 
BTC = pdr.DataReader('BTC-USD','yahoo',start_date,end_date)
BTC['ticker']= 'BTC'
ETH = pdr.DataReader('ETH-USD','yahoo',start_date,end_date)
ETH['ticker']= 'ETH'
ADA = pdr.DataReader('ADA-USD','yahoo',start_date,end_date)
ADA['ticker']= 'ADA'
BNB = pdr.DataReader('BNB-USD','yahoo',start_date,end_date)
BNB['ticker']= 'BNB'
USDT = pdr.DataReader('USDT-USD','yahoo',start_date,end_date)
USDT['ticker']= 'USDT'
DOGE = pdr.DataReader('DOGE-USD','yahoo',start_date,end_date)
DOGE['ticker']= 'DOGE'
USDC = pdr.DataReader('USDC-USD','yahoo',start_date,end_date)
USDC['ticker']= 'USDC'
MATIC = pdr.DataReader('MATIC-USD','yahoo',start_date,end_date)
MATIC['ticker']= 'MATIC'
ETC = pdr.DataReader('ETC-USD','yahoo',start_date,end_date)
ETC['ticker']= 'ETC'
BCH = pdr.DataReader('BCH-USD','yahoo',start_date,end_date)
BCH['ticker']= 'BCH'
LTC = pdr.DataReader('LTC-USD','yahoo',start_date,end_date)
LTC['ticker']= 'LTC'
LINK = pdr.DataReader('LINK-USD','yahoo',start_date,end_date)
LINK['ticker']= 'LINK'
XRP = pdr.DataReader('XRP-USD','yahoo',start_date,end_date)
XRP['ticker']= 'XRP'


# In[318]:


#fusion de toutes les tables de données 
ticker_list = [BTC, ETH, ADA, BCH, LTC, LINK, USDT, ETC, XRP]
historical_datas = BTC
for ticker in ticker_list:
       historical_datas = historical_datas.append(ticker) 


# In[320]:


#informations sur notre jeu de données 
print(historical_datas.shape)
print(historical_datas.columns)


# In[321]:


#prendre que les derniers 14 jours 
rsi_data = historical_datas.groupby('ticker').tail(14)
print(rsi_data)


# In[322]:


#création de la variable de la différence des prix entre chaque date 
hist_data = rsi_data[['ticker','Close','High','Low','Open','Volume']]
hist_data['change_in_price'] = hist_data['Close'].diff()


# In[323]:


print(hist_data)


# In[324]:


# ce code permet d'arreter le calcul de la différence des prix à chaque changement de titre. 
mask = hist_data['ticker'] != hist_data['ticker'].shift(1)
hist_data['change_in_price'] = np.where(mask == True, np.nan, hist_data['change_in_price'])
hist_data[hist_data.isna().any(axis = 1)]


# In[325]:


# création des tables up_df et down_df correspandant respectivement à l'augmentation et la diminution de la valeur
up_df, down_df = hist_data[['ticker','change_in_price']].copy(), hist_data[['ticker','change_in_price']].copy()


# In[326]:


# pour up_df : si le prix augmente on garde la difference sinon on met 0 
up_df.loc['change_in_price'] = up_df.loc[(up_df['change_in_price'] < 0), 'change_in_price'] = 0
up_df = up_df[:-1]


# In[329]:


print(up_df)


# In[330]:


# pour down_df: si le prix diminue on garde la différence sinon on met 0
down_df.loc['change_in_price'] = down_df.loc[(down_df['change_in_price'] > 0), 'change_in_price'] = 0
down_df = down_df[:-1]
print(down_df)


# In[331]:


# on met les valeurs de down_df en valeur absolue pour enlever le signe "-"
down_df['change_in_price'] = down_df['change_in_price'].abs()


# In[332]:


# calcul de la moyenne mobile exponentielle 
ewma_up = up_df.groupby('ticker')['change_in_price'].transform(lambda x: x.ewm(span = 14).mean())
ewma_down = down_df.groupby('ticker')['change_in_price'].transform(lambda x: x.ewm(span = 14).mean())
print(ewma_up.tail(14))


# In[334]:


#calcul du RSI
relative_strength = ewma_up / ewma_down
relative_strength_index = 100.0 - (100.0 / (1.0 + relative_strength))


# In[335]:


# ajout des colonnes crées au premier dataframe 
hist_data['down_days'] = down_df['change_in_price']
hist_data['up_days'] = up_df['change_in_price']
hist_data['RSI'] = relative_strength_index

# Display the head.
hist_data.head(30)


# In[336]:


#calcul de l'oscillateur stochastique ( k_percent)
low_14, high_14 = hist_data[['ticker','Low']].copy(), hist_data[['ticker','High']].copy()
low_14 = low_14.groupby('ticker')['Low'].transform(lambda x: x.rolling(window = 14).min())
high_14 = high_14.groupby('ticker')['High'].transform(lambda x: x.rolling(window = 14).max())
k_percent = 100 * ((hist_data['Close'] - low_14) / (high_14 - low_14))
hist_data['low_14'] = low_14
hist_data['high_14'] = high_14
hist_data['k_percent'] = k_percent
hist_data.head(30)


# In[337]:


#Calcul du pourcentage R de william (r_percent)
low_14, high_14 = hist_data[['ticker','Low']].copy(), hist_data[['ticker','High']].copy()
low_14 = low_14.groupby('ticker')['Low'].transform(lambda x: x.rolling(window = 14).min())
high_14 = high_14.groupby('ticker')['High'].transform(lambda x: x.rolling(window = 14).max())
r_percent = ((high_14 - hist_data['Close']) / (high_14 - low_14)) * - 100
hist_data['r_percent'] = r_percent
hist_data.head(30)


# In[338]:


#on prend la derniére valeur
final = hist_data.groupby('ticker').tail(1)
final = final[['ticker','RSI','k_percent','r_percent']]
print(final)


# In[343]:


#changement en variables catégorielles
final['RSI'] = pd.cut(final['RSI'], [0, 30, 70, 100], 
                              labels=['0-30', '30-70', '70-100'])
final['k_percent'] = pd.cut(final['k_percent'], [0, 20, 80, 100], 
                              labels=['0-20', '20-80', '80-100'])
final['r_percent'] = pd.cut(final['r_percent'], [-100, -80, -20, 0], 
                              labels=['-100/-80', '-80/-20', '-20/0'])


# In[349]:


#changement d'index
final.set_index('ticker')


# In[357]:


#application de l'algorithme k-modes et recherche du nombre de clustersc
cost = []
for num_clusters in list(range(1,5)):
    kmode = KModes(n_clusters=num_clusters, init = "Cao", n_init = 1, verbose=1)
    kmode.fit_predict(final2)
    cost.append(kmode.cost_)


# In[354]:


y = np.array([i for i in range(1,5,1)])
plt.plot(y,cost)


# In[362]:


#application du k-modes avec k = 2
from kmodes.kmodes import KModes
km_cao = KModes(n_clusters=2, init = "Cao", n_init = 1, verbose=1)
fitClusters_cao = km_cao.fit_predict(final2)


# In[361]:


# graphe des clusters 
final2_clusters = final.copy()
final2_clusters['Clusters'] = fitClusters_cao 
plt.scatter(final2_clusters['ticker'],final2_clusters['k_percent'],c=final2_clusters['Clusters'],cmap='rainbow')


# In[ ]:




