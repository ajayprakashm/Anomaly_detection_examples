# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 16:30:56 2019

@author: Admin
"""

#%% NYC Taxi Anomaly detection using autoencoder
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
from keras.layers import Dense, Input
from keras.models import Model, Sequential, load_model
from keras.callbacks import EarlyStopping, TensorBoard,ModelCheckpoint
from keras import regularizers
import seaborn as sns
import scipy.io as spio
import mplcursors

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
def autoencoder_plot(x,nplot,title):
    plt.figure()
    pred=autoencoder.predict(x)
    plt.subplot(2,1,1)
    plt.plot(x)
    plt.plot(pred)
    plt.legend(['Data','Reconstructed'])
    plt.title(title,fontsize=16,fontweight='bold')
    plt.subplot(2,1,2)
    plt.plot(np.abs(pred-x),'k')
    plt.title('Reconstruction error',fontsize=16,fontweight='bold')

#%% Seperate training and validation data, and scale the data based ONLY on 
# the "normal" data
scaler=preprocessing.StandardScaler()
x_train =scaler.fit_transform(np.reshape(x_train,(len(x_train),1)))
x_train_S1,x_test_S1=train_test_split(x_train,test_size=0.2)
x_test=scaler.transform(np.reshape(x_test,(len(x_test),1)))

#%% Keras deep network model
ncol = x_train.shape[1]
first = 100
encoding_dim = 5

input_dim = Input(shape = (x_train.shape[1], ))

# DEFINE THE ENCODER LAYERS
encoded1 = Dense(first, activation = 'tanh')(input_dim)

encoded2 = Dense(encoding_dim, activation = 'tanh')(encoded1)

# DEFINE THE DECODER LAYERS
decoded1 = Dense(first, activation = 'tanh')(encoded2)
decoded2 = Dense(ncol, activation = 'linear')(decoded1)

# COMBINE ENCODER AND DECODER INTO AN AUTOENCODER MODEL
autoencoder = Model(inputs = input_dim, outputs = decoded2)

# CONFIGURE AND TRAIN THE AUTOENCODER
autoencoder.compile(optimizer = 'SGD', loss = 'mse',metrics=['accuracy'])
#autoencoder.compile(optimizer = 'adadelta', loss = 'mean_squared_error')

AE = autoencoder.fit(x_train_S1, x_train_S1, epochs = 50, batch_size = 5, 
                shuffle = True, validation_data = (x_test_S1, x_test_S1))

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
plt.title('encoding_dim=' + str(encoding_dim))

#%% Evaluate performance on validation set of normal dataset 
autoencoder_plot(x_test,1,'Reconstructed data')
# Evaluate performance on severly faulty dataset
dat=data.values
x=preprocessing.StandardScaler().fit_transform(np.reshape(dat,(len(dat),1)))
pred=autoencoder.predict(x)
plt.figure()
ax1=plt.subplot(2,1,1)
plt.plot(data.index,x)
plt.plot(data.index,pred)
plt.title('Autoencoder Output',fontsize=12,fontweight='bold')
plt.legend(['Normal data','Reconstructed data'])
plt.xticks(rotation=0, ha='right')
date_form = DateFormatter("%d/%b/%y") # date and year
ax1.xaxis.set_major_formatter(date_form) 
ax2=plt.subplot(2,1,2,sharex=ax1)
error=np.abs(pred-x)
plt.plot(data.index,error,'k')
plt.xticks(rotation=0, ha='right')
date_form = DateFormatter("%d/%b/%y") # date and year
ax2.xaxis.set_major_formatter(date_form) 
plt.title('Reconstruction error',fontsize=12,fontweight='bold')

# isolation forest anomaly detection with contamination as 3% 
from sklearn.ensemble import IsolationForest
clf=IsolationForest(n_estimators=100,contamination=0.003)
a_train,a_test=train_test_split(error,test_size=0.3)
d=clf.fit(a_train)
ano=d.predict(error)
iso_ano_1=data[ano==-1]
iso_ano=error[ano==-1]
plt.scatter(iso_ano_1.index,iso_ano,marker='o',color='red')
plt.legend(['reconst error','Anomaly'])
mplcursors.cursor()

#%% MSE for train and test data and to deside thershold
x_test_pred = autoencoder.predict(x_test)
x_train_pred = autoencoder.predict(x_train)

mse_test = np.mean(np.abs(x_test - x_test_pred), axis=1)
mse_x_train = np.mean(np.abs(x_train - x_train_pred), axis=1)

plt.figure()
sns.distplot(mse_test,kde= True,color = 'blue')
sns.distplot(mse_x_train,kde=True,color='green')
plt.xlabel('mean squared error',fontsize=14,fontweight='bold')
plt.ylabel('density',fontsize=14,fontweight='bold')
plt.legend(['test','train','severly faulty'])
plt.title('Density Function',fontsize=16,fontweight='bold')
Threshold=np.round(4*np.std(mse_x_train),3) # 3 times standard deviation

#% MSE threshold for all plot
dat=data.values
x=preprocessing.StandardScaler().fit_transform(np.reshape(dat,(len(dat),1)))
pred=autoencoder.predict(x)
scored = pd.DataFrame()
scored['Loss_mae'] = np.mean(np.abs(pred-x), axis = 1)
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

#%% Feature Extraction
# 1 method: Extracted feature for normal dataset
encoder = Model(inputs = input_dim, outputs = encoded2)
features = encoder.predict(x_test)
features = pd.DataFrame(features, columns = ['F1', 'F2', 'F3', 'F4','F5'])
pd.plotting.scatter_matrix(features, alpha = 0.2, diagonal = 'kde')
plt.suptitle('Features from normal data',fontweight='bold',fontsize=16)

# 2 method: to extract features
hidden_representation = Sequential()
hidden_representation.add(autoencoder.layers[0])
hidden_representation.add(autoencoder.layers[1])
hidden_representation.add(autoencoder.layers[2])
features = hidden_representation.predict(x_test)

from sklearn.cluster import KMeans
clf=KMeans(n_clusters=3).fit(features) 

