# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 11:00:09 2019

@author: Admin
"""
#%% bearing anomaly using mahala score
%reset -f
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os as os
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import warnings
from sklearn import preprocessing
import seaborn as sns
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
dataset_train.plot(figsize = (12,6))

#%% Data preprocessing
scaler = preprocessing.StandardScaler()

X_train = pd.DataFrame(scaler.fit_transform(dataset_train), 
                              columns=dataset_train.columns,index=dataset_train.index)
# Random shuffle training data
X_train.sample(frac=1)

X_test = pd.DataFrame(scaler.transform(dataset_test), 
                             columns=dataset_test.columns,index=dataset_test.index)

whole_data=pd.DataFrame(scaler.transform(merged_data), 
                             columns=merged_data.columns,index=merged_data.index)

#%% calculate Mahalanobis distance
def covar_matrix(data, verbose=False):
    covariance_matrix = np.cov(data, rowvar=False)
    if is_pos_def(covariance_matrix):
        inv_covariance_matrix = np.linalg.inv(covariance_matrix)
        if is_pos_def(inv_covariance_matrix):
            return covariance_matrix, inv_covariance_matrix
        else:
            print("Error: Inverse of Covariance Matrix is not positive definite!")
    else:
        print("Error: Covariance Matrix is not positive definite!")

def MahalanobisDist(inv_cov_matrix, mean_distr, data, verbose=False):
    inv_covariance_matrix = inv_cov_matrix
    vars_mean = mean_distr
    diff = data - vars_mean
    md = []
    for i in range(len(diff)):
        md.append(np.sqrt(diff[i].dot(inv_covariance_matrix).dot(diff[i])))
    return md

def MD_detectOutliers(dist, extreme=False, verbose=False):
    k = 3. if extreme else 2.
    threshold = np.mean(dist) * k
    outliers = []
    for i in range(len(dist)):
        if dist[i] >= threshold:
            outliers.append(i)  # index of the outlier
    return np.array(outliers)

def MD_threshold(dist, extreme=False, verbose=False):
    k = 3. if extreme else 2.
    threshold = np.mean(dist) * k
    return threshold

def is_pos_def(A):
    if np.allclose(A, A.T):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False

#%% Mahala plotting
data_train = np.array(X_train.values)
data_test = np.array(X_test.values)  
ber_data= np.array(whole_data)  
cov_matrix, inv_cov_matrix  = covar_matrix(data_train)
mean_distr = data_train.mean(axis=0)

dist_test = MahalanobisDist(inv_cov_matrix, mean_distr, data_test, verbose=False)
dist_train = MahalanobisDist(inv_cov_matrix, mean_distr, data_train, verbose=False)
dist_whole = MahalanobisDist(inv_cov_matrix, mean_distr, ber_data, verbose=False)
threshold = np.round(MD_threshold(dist_train, extreme = True),2)

plt.figure()
sns.distplot(dist_train,bins = 10,kde= True, color = 'green');
sns.distplot(dist_test,bins = 10,kde= True, color = 'orange');
plt.legend(['Train','Test'])
plt.xlabel('Mahalanobis dist')

#% Mahala distance on whole dataset
anomaly_train = pd.DataFrame()
anomaly_train['Mob dist']= dist_whole
anomaly_train['Thresh'] = threshold
anomaly_train['Anomaly'] = anomaly_train['Mob dist'] > anomaly_train['Thresh']
anomaly_train.index = whole_data.index
plt.figure()
ax1=plt.axes()
ax1.set_yscale('log')
plt.plot(anomaly_train.index,anomaly_train['Mob dist'])
plt.plot(anomaly_train.index,anomaly_train['Thresh'])
plt.legend(['Mob Distance','Threshold: '+str(threshold)])
plt.xticks(rotation=0, ha='right')
date_form = DateFormatter("%d/%b/%y\n%H:%M") # date and year
ax1.xaxis.set_major_formatter(date_form) 
plt.title('Mahalanobis distance plot for bearing data')   
mplcursors.cursor()
