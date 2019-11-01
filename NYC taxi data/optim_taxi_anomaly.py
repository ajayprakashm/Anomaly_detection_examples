# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 11:59:45 2019

@author: Admin
"""

#%% NYC Taxi Anomaly detection using autoencoder 
# Optimizing hyperparmeter using gridseachcv in keras
# Importing fuction 
%reset -f
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os as os
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import scipy.io as spio
from scipy import stats
from sklearn import preprocessing
from keras.layers import Dense, Input
from keras.models import Model, Sequential, load_model
from keras.callbacks import EarlyStopping, TensorBoard,ModelCheckpoint
from keras import regularizers
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.manifold import TSNE
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

x_train,x_test=train_test_split(data,test_size=0.3)
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
x_train =preprocessing.StandardScaler().fit_transform(np.reshape(x_train,(len(x_train),1)))
x_train_S1,x_test_S1=train_test_split(x_train,test_size=0.2)
x_test=preprocessing.StandardScaler().fit_transform(np.reshape(x_test,(len(x_test),1)))

#%% Keras deep network model
ncol = x_train.shape[1]
first = 100
second = 50
encoding_dim = 5
def create_model(loss='mse'):
    input_dim = Input(shape = (x_train.shape[1], ))
    # DEFINE THE ENCODER LAYERS
    encoded1 = Dense(100, activation = 'relu')(input_dim)
    encoded2 = Dense(encoding_dim, activation = 'relu')(encoded1)

    # DEFINE THE DECODER LAYERS
    decoded1 = Dense(100, activation = 'relu')(encoded2)
    decoded2 = Dense(ncol, activation = 'relu')(decoded1)
    
    # COMBINE ENCODER AND DECODER INTO AN AUTOENCODER MODEL
    autoencoder = Model(inputs = input_dim, outputs = decoded2)
    autoencoder.compile(optimizer = 'adam', loss='mse',metrics=['accuracy'])
    #autoencoder.compile(optimizer = 'adadelta', loss = 'mean_squared_error')
    return autoencoder

model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=10, verbose=0)
loss = ['binary_crossentropy', 'mse']
param_grid = dict(loss=loss)
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid_result = grid.fit(x_train_S1, x_train_S1)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

#%%g
AE = grid.fit(x_train_S1, x_train_S1, epochs = 50, batch_size = 25, 
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
# pick a random row to plot

#plt.plot(df['2017':].asfreq('W').value, marker='.') # eth['2017':] returns a subset of eth since 2017
# Evaluate performance on slightly faulted dataset
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
ax2=plt.subplot(2,1,2,sharex=ax1)
error=np.abs(pred-x)
plt.plot(data.index,error,'k')
plt.title('Reconstruction error',fontsize=12,fontweight='bold')

# isolation forest anomaly detection with contamination as 3% 
anomaly=error[error>0.02]
Total_anomaly=data[anomaly==error]

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


#%%
from scipy import stats
stats.ttest_ind(n_pred_test,X_test)

import random
random.sample(list(np.arange(0,100)),10)
a=np.linspace(0,984,10)
int(a)


t_s=stats.ttest_ind(faulty[:-100],faulty_pred[:-100])





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


