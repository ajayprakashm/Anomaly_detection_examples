# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 12:03:25 2019

@author: Admin
"""

#%% NYC Taxi Anomaly detection using LSTM  autoencoder
# Importing fuction 
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
from keras.layers import Dense, Input, LSTM
from keras.models import Model, Sequential, load_model
from keras.callbacks import EarlyStopping, TensorBoard,ModelCheckpoint
from keras import regularizers
import seaborn as sns
import mplcursors
from sklearn.ensemble import IsolationForest

warnings.filterwarnings("ignore")
plt.close('all')
sns.set(style="whitegrid")
np.random.seed(203)
# modify the path to correspond to where your data is
os.chdir('D:/NYC taxi data/')

#%% Datasets reading and spliting into train and test
data=pd.read_csv('nyc_taxi.csv')
data.set_index("timestamp", inplace = True) 
data.index = pd.to_datetime(data.index)
data= data.sort_index()

start='2014-07-01 00:00:00'
end='2014-07-14 00:00:00'
d=data[start:end]
x_train,x_test=train_test_split(d,test_size=0.3,random_state=32)
x_train=x_train['value'].values
x_test=x_test['value'].values

#%% Functions used in this code
# This function will chunk a vector up into blocks of length cwidth, advancing the 
# window every stepsize samples. Returns a matrix with one block per row
def autoencoder_plot(x,title):
    plt.figure()
    pred=lstm_autoencoder.predict(x)
    plt.subplot(2,1,1)
    plt.plot(x[:,0])
    plt.plot(pred[:,0])
    plt.legend(['Data','Reconstructed'])
    plt.title(title,fontsize=16,fontweight='bold')
    plt.subplot(2,1,2)
    plt.plot(np.abs(pred[:,0]-x[:,0]),'k')
    plt.title('Reconstruction error',fontsize=16,fontweight='bold')

#%% Seperate training and validation data, and scale the data based ONLY on 
# the "normal" data
scaler=preprocessing.StandardScaler()
x_train =scaler.fit_transform(np.reshape(x_train,(len(x_train),1)))
x_train_S1,x_test_S1=train_test_split(x_train,test_size=0.2)
x_test=scaler.transform(np.reshape(x_test,(len(x_test),1)))

#%% LSTM Keras deep network model
# LSTM input layer is 3D(no.of samples,lookback,no. of features)
# lookback= np of rows through which result is predicted
x_train_S1 = np.reshape(x_train_S1, (x_train_S1.shape[0], 1, x_train_S1.shape[1]))
x_test_S1 = np.reshape(x_test_S1, (x_test_S1.shape[0], 1, x_test_S1.shape[1]))

from keras.layers import RepeatVector,TimeDistributed
lstm_autoencoder = Sequential()
# Encoder
lstm_autoencoder.add(LSTM(32, activation='tanh', input_shape=(1, 1), return_sequences=True))
lstm_autoencoder.add(LSTM(16, activation='tanh', return_sequences=False))
lstm_autoencoder.add(RepeatVector(1)) #no. of feature=1
# Decoder
lstm_autoencoder.add(LSTM(16, activation='tanh',return_sequences=True))
lstm_autoencoder.add(LSTM(32, activation='tanh',return_sequences=True))
lstm_autoencoder.add(TimeDistributed(Dense(1))) #no. of feature=1
lstm_autoencoder.compile(loss='mse', optimizer='adam')
lstm_autoencoder.summary()

AE = lstm_autoencoder.fit(x_train_S1, x_train_S1,epochs = 25, batch_size = 5, 
                validation_data=(x_test_S1, x_test_S1),shuffle = True)

#%% Plotting performance of autoencoder
# Visualize loss history
training_loss = AE.history['loss']
test_loss = AE.history['val_loss']

plt.figure()
plt.plot(training_loss, 'r--')
plt.plot(test_loss, 'b-')
plt.legend(['Training Loss', 'Test Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('encoding_dim=' + str(1))

#%% Evaluate performance on validation set of normal dataset 
autoencoder_plot(x_test_S1,'Reconstructed data')
# Evaluate performance on whole dataset
dat=data.values
x=preprocessing.StandardScaler().fit_transform(np.reshape(dat,(len(dat),1)))
x=np.reshape(x, (x.shape[0], 1, x.shape[1]))
plt.figure()
ax1=plt.subplot(2,1,1)
pred=lstm_autoencoder.predict(x)
plt.plot(data.index,x[:,0])
plt.plot(data.index,pred[:,0])
plt.title('LSTM Autoencoder Output',fontsize=12,fontweight='bold')
plt.legend(['Normal data','Reconstructed data'])
plt.xticks(rotation=0, ha='right')
date_form = DateFormatter("%d/%b/%y") # date and year
ax1.xaxis.set_major_formatter(date_form) 
ax2=plt.subplot(2,1,2,sharex=ax1)
error=np.abs(pred[:,0]-x[:,0])
plt.plot(data.index,error,'k')
plt.title('Reconstruction error',fontsize=12,fontweight='bold')
plt.xticks(rotation=0, ha='right')
date_form = DateFormatter("%d/%b/%y") # date and year
ax2.xaxis.set_major_formatter(date_form) 

# isolation forest anomaly detection with contamination as 3% 
clf=IsolationForest(n_estimators=100,contamination=0.003)
a_train,a_test=train_test_split(error,test_size=0.3)
d=clf.fit(a_train)
ano=d.predict(error)
iso_ano_1=data[ano==-1]
iso_ano=error[ano==-1]

plt.scatter(iso_ano_1.index,iso_ano,marker='o',color='red')
plt.legend(['reconst error','Anomaly'])
mplcursors.cursor()

#%% Calculate MSRE statistic and plotting distribution
x_train=np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
x_test_pred = lstm_autoencoder.predict(x_test)
x_train_pred = lstm_autoencoder.predict(x_train)

mse_test = np.mean(np.abs(x_test - x_test_pred), axis=1)
mse_x_train = np.mean(np.abs(x_train[:,0] - x_train_pred[:,0]), axis=1)

plt.figure()
sns.distplot(mse_x_train,kde=True,color='green')
plt.xlabel('mean squared error',fontsize=14,fontweight='bold')
plt.ylabel('density',fontsize=14,fontweight='bold')
plt.legend(['train'])
plt.title('Density Function',fontsize=16,fontweight='bold')
Threshold=np.round(4*np.std(mse_x_train),3) # 3 times standard deviation

#% MSE threshold for all plot
scored = pd.DataFrame()
scored['Loss_mae'] = np.mean(np.abs(pred[:,0]-x[:,0]), axis = 1)
scored['Threshold'] = Threshold # From dist plot
scored['Anomaly'] = scored['Loss_mae'] > scored['Threshold']
scored.index=data.index
plt.figure()
ax1=plt.axes()
plt.plot(scored.index,scored['Loss_mae'])
plt.plot(scored.index,scored['Threshold'])
plt.legend(['Loss_MSE','Threshold: '+str(Threshold)])
plt.title('Mean Squared Error on NYC taxi data')
plt.xticks(rotation=0, ha='right')
date_form = DateFormatter("%d/%b/%y\n%H:%M") # date and year
ax1.xaxis.set_major_formatter(date_form)    
mplcursors.cursor()

#%% Feature Extradaction
# Extracted feature for normal dataset
encoder = Model(inputs = input_dim, outputs = encoded3)
features = encoder.predict(x_test)
features = pd.DataFrame(features, columns = ['F1', 'F2', 'F3', 'F4','F5'])
pd.plotting.scatter_matrix(features, alpha = 0.2, diagonal = 'kde')
plt.suptitle('Features from normal data',fontweight='bold',fontsize=16)

hidden_representation = Sequential()
hidden_representation.add(autoencoder.layers[0])
hidden_representation.add(autoencoder.layers[1])
hidden_representation.add(autoencoder.layers[2])
hidden_representation.add(autoencoder.layers[3])
hidden_representation.add(autoencoder.layers[4])

features = hidden_representation.predict(x_test)


