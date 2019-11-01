# Anomaly detection on NYC taxi dataset
This dataset is downloaded from Kaggle.

## Objective
To find the anomaly in taxi data for certain events

## Prerequisites
Install following packages:
```
pip install numpy
pip install pandas
pip install sklearn
pip install keras
pip install seaborn
```
## Python Algorithm
This algorithm is written in python language. Keras deep learning autoencoder model is explained step by step.

### Step 1: Importing Library
Importing libraries and setting random seed from numpy library
```
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os as os
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from keras.layers import Dense, Input
from keras.models import Model, load_model
from keras import regularizers
import seaborn as sns
import scipy.io as spio
import mplcursors
plt.close('all')
sns.set(style="whitegrid")
np.random.seed(203)
```
### Step 2 : Data loading and dividing into training
Data is loaded from the working directory and from 01<sup>th</sup> Jul 2014 to 14<sup>th</sup> Jul 2014 23:52:39 data is used for training the deep autoencoder model.
```
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
```

### Step 3: Normalising data
Standardise data using StandardScaler such that mean=0 and standard deviation is 1. Standardising data on training data.
```
scaler=preprocessing.StandardScaler()
x_train =scaler.fit_transform(np.reshape(x_train,(len(x_train),1)))
x_train_S1,x_test_S1=train_test_split(x_train,test_size=0.2)
x_test=scaler.transform(np.reshape(x_test,(len(x_test),1)))
```
### Step4: Keras Deep Autoencoder model
Keras model is built with two hidden layers with 50 and 10 neurons, encoding dimension is kept as two, which means bearing dataset is compressed into two columns.
```
ncol = x_train.shape[1]
first = 100
encoding_dim = 3

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
autoencoder.compile(optimizer = 'SGD', loss = 'mse')
#autoencoder.compile(optimizer = 'adadelta', loss = 'mean_squared_error')

AE = autoencoder.fit(x_train_S1, x_train_S1, epochs = 50, batch_size = 5, 
                shuffle = True, validation_data = (x_test_S1, x_test_S1))

```
### Step5: Plotting performance of model
#### Training loss and validation loss plot:
<details><summary>Code</summary>
<p>
  
```python
training_loss = AE.history['loss']
test_loss = AE.history['val_loss']

plt.figure()
plt.plot(training_loss, 'r--')
plt.plot(test_loss, 'b-')
plt.legend(['Training Loss', 'Test Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('encoding_dim=' + str(encoding_dim))
```
</p>
</details>
<p align="center"> 
<img src="https://github.com/intellipredikt/Anomaly-Detection/blob/master/IMS%20Bearing/Image/Loss.png" width="600" height="400">
</p>

#### REconstruction plot:
<details><summary>Code</summary>
<p>

```python
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
``` 
</p>
</details>
<p align="center"> 
<img src="https://github.com/intellipredikt/Anomaly-Detection/blob/master/IMS%20Bearing/Image/density%20plot.png" width="600" height="400">
</p>

From the above proablilty distribuition plot thereshold is set to 3 times of standard deviation as shown in text box. This threshold willbe used to flag anomaly in the merged bearing dataset. 

<details><summary>Code</summary>
<p>

```python
whole_data=pd.DataFrame(scaler.transform(merged_data),columns=merged_data.columns,index=merged_data.index) #Normalising with same scaler 
Testing = pd.DataFrame(index=whole_data.index)
pred= model.predict(whole_data)
ber_test = np.mean(np.power(np.abs(pred-whole_data.values), 2), axis=1)
Testing['Loss_mae'] = ber_test
Testing['Threshold'] = Threshold
Testing['Anomaly'] = Testing['Loss_mae'] > Testing['Threshold']
plt.figure()
ax1=plt.axes()
ax1.set_yscale('log')
plt.plot(Testing.index,Testing['Loss_mae'])
plt.plot(Testing.index,Testing['Threshold'])
plt.legend(['Mean squared error','Threshold: '+str(Threshold)])
plt.xticks(rotation=0, ha='right')
date_form = DateFormatter("%d/%b/%y\n%H:%M") # date and year
ax1.xaxis.set_major_formatter(date_form) 
plt.title('Mean Squared Error on bearing data')   
mplcursors.cursor()
``` 
</p>
</details>
<p align="center"> 
<img src="https://github.com/intellipredikt/Anomaly-Detection/blob/master/IMS%20Bearing/Image/mse.png" width="600" height="400">
</p>

#### Reconstruction error plot:
Reconstruction error is calculated by calculating difference in mean of bearing dataset rowwise and mean of predicted dataset rowwise i.e., it is just the difference of actual and predicted.
<details><summary>Code</summary>
<p>

```python
mean_data=np.mean(whole_data.values,axis=1) 
mean_pred = np.mean(pred, axis=1)

plt.figure()
ax1=plt.subplot(2,1,1)
plt.plot(whole_data.index,mean_data)
plt.plot(whole_data.index,mean_pred)
plt.legend(['Data','Reconstructed'])
plt.title('Reconstructed plot',fontsize=16,fontweight='bold')
plt.xticks(rotation=0, ha='right')
date_form = DateFormatter("%d/%b/%y\n%H:%M") # date and year
ax1.xaxis.set_major_formatter(date_form)  
ax2=plt.subplot(2,1,2,sharex=ax1)
RE=np.abs(mean_data-mean_pred)
plt.plot(whole_data.index,RE,'k')
plt.legend(['Recontruction Error'])
plt.xticks(rotation=0, ha='right')
ax2.xaxis.set_major_formatter(date_form)
plt.tight_layout()
```
</p>
</details>
<p align="center"> 
<img src="https://github.com/intellipredikt/Anomaly-Detection/blob/master/IMS%20Bearing/Image/Reconstruction.png" width="600" height="400">
</p>

From plot it can be seen, reconstruction loss is very less till 16<sup>th</sup> Feb 2004 after that it increases drastically.

### Step 6: Dividing into Normal and Abnormal class
From the reconstruction and mean squared plot it can be clearly seen before 16<sup>th</sup> Feb 2004 data is normal and after that abnormality starts till failure.
```
S1='2004-02-12 10:32:39'
S2='2004-02-16 03:02:39'

normal=merged_data[S1:S2]
abnormal=merged_data['2004-02-16 03:02:39':]
y_normal=np.zeros(len(normal))
y_abnormal=np.ones(len(abnormal))
y=np.concatenate([y_normal,y_abnormal])

# Noramalising using same scaler
x_1= pd.DataFrame(scaler.transform(normal),columns=normal.columns,index=normal.index)
x_2= pd.DataFrame(scaler.transform(abnormal),columns=abnormal.columns,index=abnormal.index)
```

From 12<sup>th</sup> Feb 2004 10:32:39 to 16<sup>th</sup> Feb 2004 03:02:39 data are considered as normal and from 16<sup>th</sup> Feb 2004 03:12:39 till failure is considered as abnormal data. 

#### Autoencoder features
Two latent features are extracted from autoencoder model for normal & abnormal data and scatter plot is plotted for the two features.
<details><summary>Code</summary>
<p>

```python
hidden_representation = Sequential()
hidden_representation.add(model.layers[0])
hidden_representation.add(model.layers[1])
hidden_representation.add(model.layers[2])
hidden_representation.add(model.layers[3])

# Visualizing normal air filter data
feat_x_1= hidden_representation.predict(x_1)
feat_x_2= hidden_representation.predict(x_2)

plt.scatter(feat_x_1[:,0],feat_x_1[:,1],alpha=0.5)
plt.scatter(feat_x_2[:,0],feat_x_2[:,1],alpha=0.5)
plt.xlabel('1st Feature')
plt.ylabel('2nd Feature')
plt.legend(['Normal (2004-02-12 10:32:39 to 2004-02-16 03:02:39)',
            'Abnormal (2004-02-16 03:12:39 to 2004-02-19 06:22:39)'])
plt.title('Deep Autoencoder with 2 features')
``` 
</p>
</details>
<p align="center"> 
<img src="https://github.com/intellipredikt/Anomaly-Detection/blob/master/IMS%20Bearing/Image/classification.png" width="600" height="400">
</p>
From the above plot it can be seen that autoencoder features are able to classify normal and abnormal class very effectively.

#### TSNE plot:
TSNE scatter plot is plotted for first two components to get clear representation for normal and abnormal class.
<details><summary>Code</summary>
<p>

```python
def tsne_plot(x1, y1):
    tsne = TSNE(n_components=2, random_state=0)
    X_t = tsne.fit_transform(x1)
    plt.figure(figsize=(12, 8))
    plt.scatter(X_t[np.where(y1 == 0), 0], X_t[np.where(y1 == 0), 1], marker='o', color='g', linewidth='1', alpha=0.8, label='normal')
    plt.scatter(X_t[np.where(y1 == 1), 0], X_t[np.where(y1 == 1), 1], marker='o', color='r', linewidth='1', alpha=0.8, label='25%       Blocked') 
    plt.legend(loc='best')
    plt.legend(['Normal','Abnormal'])
    plt.title('TSNE plot',fontsize=20)

rep_x=pd.concat([normal,abnormal])
rep_y = y
tsne_plot(rep_x, rep_y)
```
</p>
</details>
<p align="center"> 
<img src="https://github.com/intellipredikt/Anomaly-Detection/blob/master/IMS%20Bearing/Image/TSNE.png" width="600" height="400">
</p>

## Conclusion
Autoencoder features can be used for classification into different class. This study is done using two layers, it can be more for clear classification. Optimization of parameters like activation function, layers, number of neuron etc., is not done in this algorithm.

