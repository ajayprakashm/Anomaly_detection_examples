# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 11:06:40 2019

@author: ajay
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
from scipy import stats
from sklearn import preprocessing
from keras.layers import Dense, Input
from keras.models import Model, Sequential, load_model
from keras.models import model_from_json
from sklearn.manifold import TSNE
import seaborn as sns
import scipy.io as spio
import mplcursors

warnings.filterwarnings("ignore")
plt.close('all')
sns.set(style="whitegrid")
# modify the path to correspond to where your data is
os.chdir('D:/NASA bearing dataset')
merged_data=pd.read_csv('merged_dataset_BearingTest_1.csv')
merged_data.index=pd.to_datetime(merged_data[merged_data.columns[0]])
merged_data=merged_data.drop(merged_data.columns[0],axis=1)
data_train = merged_data['2004-02-12 11:02:39':'2004-02-14 23:52:39']

#%% normalising data and fit on data_train
scaler = preprocessing.StandardScaler().fit(data_train)

x_train= pd.DataFrame(scaler.transform(data_train),columns=data_train.columns, 
                              index=data_train.index)

#%% Keras deep autoencoder model
np.random.seed(203)
ncol = x_train.shape[1]
first = 50
second= 10
encoding_dim = 2

input_dim = Input(shape = (ncol, ))

# DEFINE THE ENCODER LAYERS
encoded1 = Dense(first, activation = 'relu')(input_dim)
encoded2 = Dense(second, activation = 'relu')(encoded1)

encoded3 = Dense(encoding_dim, activation = 'relu')(encoded2)

# DEFINE THE DECODER LAYERS
decoded1 = Dense(second, activation = 'relu')(encoded3)
decoded2 = Dense(first, activation = 'relu')(decoded1)

decoded3 = Dense(ncol, activation = 'linear')(decoded2)

model = Model(inputs = input_dim, outputs = decoded3)
model.compile(optimizer = 'SGD', loss = 'mean_squared_error',metrics=['accuracy'])

history = model.fit(x_train, x_train, epochs = 200, batch_size = 5, 
                shuffle = True, validation_split=0.1)

#% plotting training error
plt.plot(history.history['loss'],'b',label='Training loss')
plt.plot(history.history['val_loss'],'r',label='Validation loss')
plt.legend(loc='upper right')
plt.title('Model loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

#%% Calculate MSRE statistic and plotting distribution
x_train_pred = model.predict(x_train)

mse_x_train = np.mean(np.power(np.abs(x_train_pred-x_train), 2), axis=1)

plt.figure()
sns.distplot(mse_x_train,kde=True,color='green')
plt.xlabel('mean squared error',fontsize=14,fontweight='bold')
plt.ylabel('density',fontsize=14,fontweight='bold')
plt.legend(['train'])
plt.title('Density Function',fontsize=16,fontweight='bold')
Threshold=np.round(4*np.std(mse_x_train),3) # 3 times standard deviation

# anomaly threshold on whole bearing dataset
whole_data=pd.DataFrame(scaler.transform(merged_data),columns=merged_data.columns, 
                             index=merged_data.index)
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

#%% Reconstruction Error
# Data is compresed into one by taking mean
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
mplcursors.cursor()
plt.tight_layout()

#%% Dividing into two class normal and abnormal and plotting scatter plot of
# two autoencoder features
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

# hidden representation of bearing dataset
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

# =============================================================================
# # 3D scatter plot
# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(feat_norm[:,0], feat_norm[:,1], feat_norm[:,2], c='r', marker='o',alpha=0.2)
# ax.scatter(feat_test[:,0], feat_test[:,1], feat_test[:,2], c='y', marker='o')
# 
# =============================================================================
#%% TSNE plot
def tsne_plot(x1, y1):
    tsne = TSNE(n_components=2, random_state=0)
    X_t = tsne.fit_transform(x1)
    plt.figure(figsize=(12, 8))
    plt.scatter(X_t[np.where(y1 == 0), 0], X_t[np.where(y1 == 0), 1], marker='o', color='g', linewidth='1', alpha=0.8, label='normal')
    plt.scatter(X_t[np.where(y1 == 1), 0], X_t[np.where(y1 == 1), 1], marker='o', color='r', linewidth='1', alpha=0.8, label='25% Blocked') 
    plt.legend(loc='best')
    plt.legend(['Normal','Abnormal'])
    plt.title('TSNE plot',fontsize=20)

rep_x=pd.concat([normal,abnormal])
rep_y = y
tsne_plot(rep_x, rep_y)

#%% SVM for classification into normal and abnormal
from sklearn import svm
from sklearn import metrics
import pickle

bear_fea=np.concatenate([feat_x_1,feat_x_2])
train_fea,test_fea,y_train_fea,y_test_fea=train_test_split(bear_fea,y,test_size=0.3,random_state=32)
def classifier(clf,x):
    pred=clf.predict(x)
    prob=clf.predict_log_proba(x)
    label=[]
    for i in np.arange(0,len(pred)):
        if pred[i]==0:
            label.append('Normal')
        else:
            label.append('Abnormal')           
    return pred,label,prob

# SVC training
clf=svm.SVC(kernel='rbf',probability=True).fit(train_fea,y_train_fea) 
predict,label,prob=classifier(clf,test_fea)
print('Accuracy Score:',metrics.accuracy_score(y_test_fea,predict))
print('Confusion Matrix:\n',metrics.confusion_matrix(y_test_fea,predict))    
print('Predicted Label:',label)

#%% Saving autoencoder model with json format
# serialize model to JSON
from keras.models import load_model

os.chdir('D:/NASA bearing dataset/model')
model.save('IMS_bearing.h5')
print("Saved model to disk")

#% save SVM classiifer model to disk
m=np.mean(data_train)
s=np.std(data_train)
pickle.dump(clf, open('SVM_Classifier.clf', 'wb'))
pickle.dump(m, open('mean', 'wb'))
pickle.dump(s, open('std', 'wb'))

#%% Running loaded model from disk for test data
# Importing library
from keras.models import load_model
from keras.models import Sequential
import numpy as np
import pandas  as pd
import pickle
import os
# loading autoencoder model
os.chdir('D:/NASA bearing dataset/model')
model = load_model('IMS_bearing.h5')

# loading SVM Classifier, mean and std
clf = pickle.load(open('SVM_Classifier.clf', 'rb'))
m=pickle.load(open('mean','rb'))
s=pickle.load(open('std','rb'))

# Classifier function
def classifier(clf,x):
    pred=clf.predict(x)
    prob=clf.predict_proba(x)
    label=[]
    for i in np.arange(0,len(pred)):
        if pred[i]==0:
            label.append('Normal')
        else:
            label.append('Abnormal')           
    return pred,label,prob

#%% Running saved model
hidden_representation = Sequential()
hidden_representation.add(model.layers[0])
hidden_representation.add(model.layers[1])
hidden_representation.add(model.layers[2])
hidden_representation.add(model.layers[3])

TD=np.array([merged_data.iloc[1,:].values]).reshape((-1,4))
nor_td=(merged_data-m)/s # Standarising with respect to traing data
#nor_td=(TD-np.reshape(m.values,(-1,4)))/np.reshape(s.values,(-1,4)) # Standarising with respect to traing data

# Reconstruction Error
mean_data=np.mean(nor_td,axis=1) 
mean_pred = np.mean(model.predict(nor_td), axis=1)
plt.figure()
ax1=plt.subplot(2,1,1)
plt.plot(merged_data.index,mean_data)
plt.plot(merged_data.index,mean_pred)
plt.legend(['Data','Reconstructed'])
plt.title('Reconstructed plot',fontsize=16,fontweight='bold')
plt.xticks(rotation=0, ha='right')
date_form = DateFormatter("%d/%b/%y\n%H:%M") # date and year
ax1.xaxis.set_major_formatter(date_form)  
ax2=plt.subplot(2,1,2,sharex=ax1)
RE=np.abs(mean_data.values-mean_pred)
plt.plot(merged_data.index,RE,'k')
plt.legend(['Recontruction Error'])
plt.xticks(rotation=0, ha='right')
ax2.xaxis.set_major_formatter(date_form)
mplcursors.cursor()
plt.tight_layout()

# autoencoder output into two features
test_data=hidden_representation.predict(nor_td)
#predict,label,prob=classifier(clf,test_data)

def zscore(error):
    # Zscore is calculated based on reconstruction error
    mu,sigma=np.mean(np.mean(x_train,axis=1)),np.mean(np.std(x_train,axis=1))
    z_s=(error-mu)/sigma
    return z_s

z_score=zscore(RE)
label=[]
for i in np.arange(0,len(z_score)):
    if z_score[i]<=3:
        label.append(1)
    elif z_score[i]>3 and z_score[i]<=5:
        label.append(2)
    else:
        label.append(3)

plt.plot(merged_data.index,z_score)

# Saving in dataframe
output=pd.DataFrame()
output['Feature_1']=test_data[:,0]
output['Feature_2']=test_data[:,1]
output['Ber_data']=mean_data.values
output['Rec_data']=mean_pred
output['Z_Score']=z_score
output['Pred label']=label
output.index=merged_data.index
output.to_excel('Output.xlsx')

# =============================================================================
# print('Predicted Label:',label)
# print('Probability of Normal: '+"{:.1%}".format(prob[0,0]))
# print('Probability of Abnormal: '+"{:.1%}".format(prob[0,1]))
# =============================================================================

#%% DBSCAN classification
from sklearn.cluster import DBSCAN
from sklearn import metrics

db = DBSCAN(eps=0.4, min_samples=10).fit(test_data)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)


unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = test_data[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = test_data[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()

#%% Kmeans
from sklearn.cluster import KMeans
clf=KMeans(n_clusters=6).fit(test_data)
km=clf.fit_predict(test_data)
for k in np.arange(0,4):
    
    xy = test_data[km==k]
    plt.plot(xy[:, 0], xy[:, 1], 'o',
             markeredgecolor='k', markersize=5)
    
#%% Isoation Forest
from sklearn.ensemble import IsolationForest
clf=IsolationForest(max_samples=100).fit(test_data)
km=clf.fit_predict(test_data)
for k in np.unique(km):
    
    xy = test_data[km==k]
    plt.plot(xy[:, 0], xy[:, 1], 'o',
             markeredgecolor='k', markersize=5)
#%%
sns.distplot(RE)
rm=np.mean(RE)
rs=np.std(RE)
th1=1.5*rs
th2=3*rs
recon=pd.DataFrame()
recon['RE']=RE
recon['1st Thre']=RE>th1 & RE<th2

x_train_pred = model.predict(x_train)

mse_x_train = np.mean(np.abs(x_train_pred-x_train), axis=1)
plt.figure()
sns.distplot(mse_x_train,kde=True,color='green')
plt.xlabel('mean squared error',fontsize=14,fontweight='bold')
plt.ylabel('density',fontsize=14,fontweight='bold')
plt.legend(['train'])
plt.title('Density Function',fontsize=16,fontweight='bold')
Threshold=max(mse_x_train)

# anomaly threshold on whole bearing dataset
whole_data=pd.DataFrame(scaler.transform(merged_data),columns=merged_data.columns, 
                             index=merged_data.index)
Testing = pd.DataFrame(index=whole_data.index)
pred= model.predict(whole_data)
ber_test = np.mean(np.abs(pred-whole_data.values), axis=1)
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

#%%
from scipy.stats import norm
mean = np.mean(np.mean(x_train,axis=1))
stdev = np.std(np.mean(x_train,axis=1))
# calculate the pdf
pdf=norm.pdf(mean_data,mean,stdev)
# pdf = norm.logpdf(mean_data,mean,stdev)
# plot
plt.plot(pdf)

def normalize(probs):
    prob_factor = 1 / sum(probs)
    return [prob_factor * p for p in probs]

c=normalize(pdf)
plt.plot(c)

m=np.mean(RE[:366])
s=np.std(RE[:366])
z=(RE-m)/s
plt.plot(z)


#%% Rolling window
def anomalyScores(originalDF, predDF):
    loss = np.sum((np.array(originalDF)-np.array(predDF))**2, axis=1)
    loss = pd.Series(data=loss,index=originalDF.index)
    loss = (loss-np.min(loss))/(np.max(loss)-np.min(loss))
    return loss

def zscore(error):
    # Zscore is calculated based on reconstruction error
    x=np.mean(error,axis=1)
    mu,sigma=np.mean(np.mean(x_train,axis=1)),np.mean(np.std(x_train,axis=1))
    z_s=(x-mu)/sigma
    return z_s

def per_change(originalDF,error):
    # perchange wrt original
    per_chg=(np.mean(error,axis=1)/np.mean(originalDF,axis=1).values)*100 
    return per_chg

def coeff_verif(error):
    # coeff of varification is calculated based on reconstruction error
    x=np.mean(error,axis=1)
    mu,sigma=np.mean(x),np.std(x)
    CV=sigma/mu
    return CV

def anomaly(ws):
    f=plt.figure()
    k=0
    df=pd.DataFrame()
    anamoly_score=[]
    z_score=[]
    p_change=[]
    co_ver=[]
    for i in np.arange(0,int(len(mean_data)/ws)+1):
        originalDF = nor_td[k:k+ws]
        predDF = pd.DataFrame(model.predict(nor_td[k:k+ws]))
        predDF.index=originalDF.index      
        error=originalDF.values-predDF.values     
        anamoly_score.append(anomalyScores(originalDF, predDF))
        z_score.append(zscore(error))
        p_change.append(per_change(originalDF,error))
        co_ver.append(coeff_verif(error))
        ax1=plt.subplot(1,1,1)
        plt.plot(originalDF.index,z_score[i])        
        plt.legend(['Anomaly Score'])
        plt.title('Anomaly Score plot',fontsize=16,fontweight='bold')
        plt.xticks(rotation=0, ha='right')
        date_form = DateFormatter("%d/%b/%y\n%H:%M") # date and year
        ax1.xaxis.set_major_formatter(date_form) 
        k=k+ws-1
    return anamoly_score

df=anomaly(100)

#%% Prophet fb anomaly package
from fbprophet import Prophet
m = Prophet(interval_width = 1)
# Using the training data from "healthy part"
df=pd.DataFrame()
df['y']=RE[:366]
df['ds']=merged_data.index[:366]

m.fit(df)
forecast = m.predict(df)
forecast['fact'] = df['y'].reset_index(drop = True)
print('Displaying Prophet plot')
fig1 = m.plot(forecast)
fig1 = plt.plot(forecast['ds'],RE[:366])
plt.title("Fit of Training Data")

# prediction on testing data
df_test=pd.DataFrame()
df_test['y']=RE
df_test['ds']=merged_data.index

forecast_pr = m.predict(df_test)
forecast_pr['fact'] = df_test['y'].reset_index(drop = True)
print('Displaying Prophet plot')
fig1 = m.plot(forecast_pr)
fig1 = plt.plot(forecast_pr['ds'],RE)

def detect_anomalies(forecast):
    forecasted = forecast[['ds','trend', 'yhat', 'yhat_lower', 'yhat_upper', 'fact']].copy()
    #forecast['fact'] = df['y']

    forecasted['anomaly'] = 0
    forecasted.loc[forecasted['fact'] > forecasted['yhat_upper'], 'anomaly'] = 1
    forecasted.loc[forecasted['fact'] < forecasted['yhat_lower'], 'anomaly'] = -1

    #anomaly importances
    forecasted['importance'] = 0
    forecasted.loc[forecasted['anomaly'] ==1, 'importance'] = \
        (forecasted['fact'] - forecasted['yhat_upper'])/forecast['fact']
    forecasted.loc[forecasted['anomaly'] ==-1, 'importance'] = \
        (forecasted['yhat_lower'] - forecasted['fact'])/forecast['fact']
    
    return forecasted

pred = detect_anomalies(forecast)