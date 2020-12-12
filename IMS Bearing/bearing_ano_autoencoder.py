# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 11:21:53 2019

@author: Ajay
"""

#%% bearing anomaly using autoencoder mse
%reset -f
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os as os
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import warnings
from sklearn.preprocessing import StandardScaler
import scipy.io as spio
from scipy import stats
from sklearn import preprocessing
from keras.layers import Dense, Input
from keras.models import Model, Sequential, load_model
from keras import regularizers
from sklearn.manifold import TSNE
import seaborn as sns
import scipy.io as spio
import mplcursors

warnings.filterwarnings("ignore")
plt.close('all')
sns.set(style="whitegrid")
np.random.seed(203)
# modify the path to correspond to where your data is
os.chdir('D:/NASA bearing dataset')
merged_data=pd.read_csv('merged_dataset_BearingTest_1.csv')
merged_data.index=pd.to_datetime(merged_data[merged_data.columns[0]])
merged_data=merged_data.drop(merged_data.columns[0],axis=1)
dataset_train = merged_data['2004-02-12 11:02:39':'2004-02-13 23:52:39']
dataset_test = merged_data['2004-02-13 23:52:39':]

#%% Data preprocessing
scaler = preprocessing.MinMaxScaler()

x_train = pd.DataFrame(scaler.fit_transform(dataset_train), 
                              columns=dataset_train.columns, 
                              index=dataset_train.index)
# Random shuffle training data
x_train.sample(frac=1)

x_test = pd.DataFrame(scaler.transform(dataset_test), 
                             columns=dataset_test.columns, 
                             index=dataset_test.index)
whole_data=pd.DataFrame(scaler.transform(merged_data), 
                             columns=merged_data.columns, 
                             index=merged_data.index)
#%%
np.random.seed(10)
act_func = 'relu'

# Input layer:
model=Sequential()
# First hidden layer, connected to input vector X. 
model.add(Dense(10,activation=act_func,
                kernel_initializer='glorot_uniform',
                kernel_regularizer=regularizers.l2(0.0),
                input_shape=(x_train.shape[1],)))
model.add(Dense(2,activation=act_func,
                kernel_initializer='glorot_uniform'))

model.add(Dense(10,activation=act_func,
                kernel_initializer='glorot_uniform'))

model.add(Dense(x_train.shape[1],
                kernel_initializer='glorot_uniform'))

model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])

# Train model for 100 epochs, batch size of 10: 
NUM_EPOCHS=100
BATCH_SIZE=10

history=model.fit(np.array(x_train),np.array(x_train),batch_size=BATCH_SIZE, 
                  epochs=NUM_EPOCHS,
                  validation_split=0.05,
                  verbose = 1)
#%% plotting training error
plt.plot(history.history['loss'],'b',label='Training loss')
plt.plot(history.history['val_loss'],'r',label='Validation loss')
plt.legend(loc='upper right')
plt.xlabel('Epochs')
plt.ylabel('Loss, [mse]')
plt.ylim([0,.1])
plt.show()

#%% Calculate MSRE statistic and plotting distribution
x_test_pred = model.predict(x_test)
x_train_pred = model.predict(x_train)

mse_test = np.mean(np.power(np.abs(x_test_pred-x_test), 2), axis=1)
mse_x_train = np.mean(np.power(np.abs(x_train_pred-x_train), 2), axis=1)

plt.figure()
#sns.distplot(mse_test,kde= True,color = 'blue')
sns.distplot(mse_x_train,kde=True,color='green')
plt.xlabel('mean squared error',fontsize=14,fontweight='bold')
plt.ylabel('density',fontsize=14,fontweight='bold')
plt.legend(['train'])
plt.title('Density Function',fontsize=16,fontweight='bold')
Threshold=np.round(4*np.std(mse_x_train),3) # 3 times standard deviation

# anomaly threshold on whole bearing dataset
scored_test = pd.DataFrame(index=whole_data.index)
pred= model.predict(whole_data)
ber_test = np.mean(np.power(np.abs(pred-whole_data.values), 2), axis=1)
scored_test['Loss_mae'] = ber_test
scored_test['Threshold'] = Threshold
scored_test['Anomaly'] = scored_test['Loss_mae'] > scored_test['Threshold']
plt.figure()
ax1=plt.axes()
ax1.set_yscale('log')
plt.plot(scored_test.index,scored_test['Loss_mae'])
plt.plot(scored_test.index,scored_test['Threshold'])
plt.legend(['Loss_MSE','Threshold: '+str(Threshold)])
plt.xticks(rotation=0, ha='right')
date_form = DateFormatter("%d/%b/%y\n%H:%M") # date and year
ax1.xaxis.set_major_formatter(date_form) 
plt.title('Mean Squared Error on testing data')   
mplcursors.cursor()

