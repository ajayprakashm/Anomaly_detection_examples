# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 10:09:53 2019

@author: ajay
"""
#%%
%reset -f
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os as os
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

#Reading files
os.chdir('D:/credit card anomaly')
df=pd.read_csv('creditcard.csv')
cn=list(df.columns)
#%% K means
# =============================================================================
# from sklearn import preprocessing 
# from sklearn.cluster import KMeans
# 
# cl=KMeans(n_clusters=2).fit(x)
# pre=cl.predict(x)
# plt.scatter(x[:,0][pre==0],x[:,1][pre==0])
# plt.scatter(x[:,0][pre==1],x[:,1][pre==1])
# =============================================================================

#%% Keras 
## input layer
from sklearn import preprocessing
from keras.layers import Dense, Input, Conv2D, LSTM, MaxPool2D, UpSampling2D
from keras.models import Model, Sequential, load_model
from keras.callbacks import EarlyStopping, TensorBoard,ModelCheckpoint
from keras import regularizers
from sklearn.manifold import TSNE
import seaborn as sns

# For plotting 
def tsne_plot(x1, y1, name="graph.png"):
    tsne = TSNE(n_components=2, random_state=0)
    X_t = tsne.fit_transform(x1)
    plt.figure(figsize=(12, 8))
    plt.scatter(X_t[np.where(y1 == 0), 0], X_t[np.where(y1 == 0), 1], marker='o', color='g', linewidth='1', alpha=0.8, label='Normal')
    plt.scatter(X_t[np.where(y1 == 1), 0], X_t[np.where(y1 == 1), 1], marker='o', color='r', linewidth='1', alpha=0.8, label='Faulty')
    plt.legend(loc='best');
    #plt.savefig(name);
    plt.show();

data=df[cn[1:29]]
label=df[cn[30]]

x_train,x_test,y_train,y_test=train_test_split(data,label,test_size=0.3,random_state=32)
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# correlation among variables
corrmat=data.corr()
f, ax = plt.subplots(figsize =(9, 8)) 
sns.heatmap(corrmat, ax = ax, cmap ="YlGnBu", linewidths = 0.1)
 
#PCA 
from sklearn.decomposition import pca
xtr=pca.PCA(n_components=2)
xpca=xtr.fit_transform(x_train)
da=xpca[y_train==1]
plt.scatter(da[:,0],da[:,1])
da=xpca[y_train==0]
plt.scatter(da[:,0],da[:,1],alpha=0.3)
plt.legend(['fraud','good'])

#Deep Autoencoder with 3 dense layer
normal=data[label==0]
fraud=data[label==1]
input_layer = Input(shape=(normal.shape[1],))

## encoding architecture
encode_layer1 = Dense(1000, activation='relu')(input_layer)
encode_layer2 = Dense(500, activation='relu')(encode_layer1)
encode_layer3 = Dense(28, activation='relu')(encode_layer2)

## decoding architecture
decode_layer1 = Dense(28, activation='relu')(encode_layer3)
decode_layer2 = Dense(500, activation='relu')(decode_layer1)
decode_layer3 = Dense(1000, activation='relu')(decode_layer2)

## output layer
output_layer  = Dense((normal.shape[1]))(decode_layer3)

# =============================================================================
# x_train= x_train.reshape(-1, 29)
# y_train = y_train.reshape(-1, 29)
# =============================================================================

autoencoder = Model(input_layer, output_layer)
autoencoder.compile(optimizer="adadelta", loss="mse")

# =============================================================================
# x_norm = preprocessing.MinMaxScaler().fit_transform(X.values)
# x_faulty = preprocessing.MinMaxScaler().fit_transform(pd.DataFrame(d3).values) # Severe faultyh data is used for prediction
# =============================================================================

# First 10000 samples is used from normal data because of time constrain
autoencoder.fit(normal[:1000], normal[:1000],batch_size = 25, epochs = 100, 
                shuffle = True, validation_split = 0.20)

hidden_representation = Sequential()
hidden_representation.add(autoencoder.layers[0])
hidden_representation.add(autoencoder.layers[1])
hidden_representation.add(autoencoder.layers[2])
hidden_representation.add(autoencoder.layers[3])

norm_hid_rep = hidden_representation.predict(normal[:1000])
faulty_hid_rep = hidden_representation.predict(fraud)

#%%
# calculating MSE or reconstruction error
mse = np.mean(np.power(normal[:1000].values - norm_hid_rep, 2), axis=1)
plt.figure(1)
plt.subplot(2,1,1)
plt.plot(norm_hid_rep[:,0]*-1)
c=normal[:1000].values
plt.plot(c[:,0])
plt.legend(['autoencoded','good data'])
plt.subplot(2,1,2)
plt.plot(mse)
plt.legend(['reconstruction error'])

plt.figure(2)
mse_f= np.mean(np.power(fraud - faulty_hid_rep, 2), axis=1)
plt.subplot(2,1,1)
plt.plot(faulty_hid_rep[:,0])
plt.plot(x_test)
plt.legend(['autoencoded','bad data'])
plt.subplot(2,1,2)
plt.plot(mse_f)
plt.legend(['reconstruction error'])

#%% Ploting to visualiuze results
rep_x = np.append(norm_hid_rep, faulty_hid_rep, axis = 0)
y_g = np.zeros(norm_hid_rep.shape[0])
y_f = np.ones(faulty_hid_rep.shape[0])
rep_y = np.append(y_g, y_f)
tsne_plot(rep_x, rep_y, "comparision plot.png")

