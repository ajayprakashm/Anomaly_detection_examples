# Anomaly detection on NASA Bearing dataset
This dataset is downloaded from http://data-acoustics.com/measurements/bearing-faults/bearing-4/. Read the readme Document for IMS Bearing Data for further information on the experiment and available data.

Dataset consists of four bearing data merged into single merged dataset file. All four bearing are tested to run to failure and during test bearing 2 failed.

## Objective
To classify data into normal and abnormal data. 

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
from keras.models import Model, Sequential, load_model
from keras.models import model_from_json
from sklearn.manifold import TSNE
import seaborn as sns
np.random.seed(203)
```
### Step 2 : Data loading and dividing into training
Data is loaded from the working directory and from 12<sup>th</sup> Feb 2004 11:02:39 to 14<sup>th</sup> Feb 2004 23:52:39 data is used for training the deep autoencoder model.
```
merged_data=pd.read_csv('merged_dataset_BearingTest_1.csv')
merged_data.index=pd.to_datetime(merged_data[merged_data.columns[0]])
merged_data=merged_data.drop(merged_data.columns[0],axis=1)
data_train = merged_data['2004-02-12 11:02:39':'2004-02-14 23:52:39']
```

### Step 3: Normalising data
Standardise data using StandardScaler such that mean=0 and standard deviation is 1. Standardising data on training data.
```
scaler = preprocessing.StandardScaler().fit(data_train)

x_train= pd.DataFrame(scaler.fit(data_train),columns=data_train.columns, 
                              index=data_train.index)
```
### Step4: Keras Deep Autoencoder model
Keras model is built with two hidden layers with 50 and 10 neurons, encoding dimension is kept as two, which means bearing dataset is compressed into two columns.
```
ncol = x_train.shape[1]
first = 50
second= 10
encoding_dim = 2

input_dim = Input(shape = (ncol, ))

# Encoding layers=2
encoded1 = Dense(first, activation = 'linear')(input_dim)
encoded2 = Dense(second, activation = 'relu')(encoded1)

# Latent space=1
encoded3 = Dense(encoding_dim, activation = 'relu')(encoded2)

# Decoding layers=2
decoded1 = Dense(second, activation = 'relu')(encoded3)
decoded2 = Dense(first, activation = 'relu')(decoded1)
decoded3 = Dense(ncol, activation = 'linear')(decoded2)

model = Model(inputs = input_dim, outputs = decoded3)
model.compile(optimizer = 'SGD', loss = 'mean_squared_error',metrics=['accuracy'])
history = model.fit(x_train, x_train, epochs = 200, batch_size = 5, 
                shuffle = True, validation_split=0.1)
```
### Step5: Plotting performance of model
#### Training loss and validation loss plot:
<details><summary>Code</summary>
<p>
  
```python
plt.plot(history.history['loss'],'b',label='Training loss')
plt.plot(history.history['val_loss'],'r',label='Validation loss')
plt.legend(loc='upper right')
plt.xlabel('Epochs')
plt.ylabel('Loss, [mse]')
plt.show()
```
</p>
</details>
<p align="center"> 
<img src="https://github.com/intellipredikt/Anomaly-Detection/blob/master/IMS%20Bearing/Image/Loss.png" width="600" height="400">
</p>

#### Proabability density plot for training data:
<details><summary>Code</summary>
<p>

```python
x_train_pred = model.predict(x_train)
mse_x_train = np.mean(np.power(np.abs(x_train_pred-x_train), 2), axis=1)

plt.figure()
sns.distplot(mse_x_train,kde=True,color='green')
plt.xlabel('mean squared error',fontsize=14,fontweight='bold')
plt.ylabel('density',fontsize=14,fontweight='bold')
plt.legend(['train'])
plt.title('Density Function',fontsize=16,fontweight='bold')
Threshold=np.round(3*np.std(mse_x_train),3) # 3 times standard deviation
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

