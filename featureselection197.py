# -*- coding: utf-8 -*-
"""
Created on Thu May 30 18:34:06 2019

@author: Rachel
"""

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve
#from sklearn import metrics
#from sklearn import datasets 
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn import preprocessing
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from sklearn.metrics import accuracy_score
#####
from sklearn.feature_selection import VarianceThreshold

import os
cwd = os.getcwd()
print(cwd)

data = pd.read_csv('final12_246.csv', header=0)

data.info()
data.head()
data.columns

#############rachel copy  from here
#count click_outs
(data['click_out'] == 1).sum()
(data['click_out'] == 0).sum()



###Balancing
# Separate majority and minority classes
	
from sklearn.utils import resample

df_majority = data[data.click_out==1]
df_minority = data[data.click_out==0]

# Downsample majority class
df_majority_downsampled = resample(df_majority,  
                                 replace=False,  
                                 n_samples=255453 )    

# Combine minority class with downsampled majority class
df_downsampled = pd.concat([df_majority_downsampled, df_minority])
df_downsampled
# Display new class counts
df_downsampled.click_out.value_counts()
df_downsampled.click_out.value_counts()


########rachel copy ends here
                                
#determining which columns are part of data and which is the prediction. Removed the user_id and session_id.
x = data.loc[:,'mobile':'duration_sec']
y = data.loc[:,'click_out']

#number of futures before feature selection
len(x)
np.size(x,1)

selector = VarianceThreshold(threshold=0.9)
x=selector.fit_transform(x)
x


#split data sets into training and testing
x=preprocessing.scale(x)
test_size = 0.5
x_train, x_test, y_train, y_test = train_test_split (x,y, test_size=0.5, random_state=1)

##number of fuatures after feature selection
len(x_train)
np.size(x_train,1)


#choosing a K Value
'''
error_rate = []

for i in range(1,6):
    
    clfNN = KNeighborsClassifier(n_neighbors=i)
    clfNN.fit(x_train,y_train)
    pred_i= clfNN.predict(x_test)
    error_rate.append(np.mean(pred_i != y_test))

plt.figure(1,figsize=(10,5))
plt.plot(range(1,6),error_rate,color='blue',linestyle='dashed',
          marker='o',markerfacecolor='red',markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
'''

#Define decision Tree
clfDT =  tree.DecisionTreeClassifier()

#Define Support vector machine
#kernel='rbf', or 'poly'
#degree: refers to the degree of the polynomial kernel
clfSVMr= svm.SVC(C=100.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovo', degree=1, gamma='auto', kernel='rbf',
    max_iter=-1, random_state=None, shrinking=True,
    tol=0.001, verbose=False, probability=True)

clfSVMp= svm.SVC(C=100.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovo', degree=1, gamma='auto', kernel='poly',
    max_iter=-1, random_state=None, shrinking=True,
    tol=0.001, verbose=False, probability=True)

#Define a Naive Bayes
clfNB = GaussianNB()

#Define a Nearest Neighbours classifiers 
#n_neighbors: is the number of neighbors
#metric: is the distance measure,
clfNN = KNeighborsClassifier(n_neighbors=3)

#random Forest
clfRF = RandomForestClassifier(n_estimators=5, max_depth=5)

clfRFmd = RandomForestClassifier(n_estimators=2, max_depth=2, random_state=None)
 
#ADA Boost
clfADA = AdaBoostClassifier(n_estimators=8, learning_rate=1)

#ANN Classifier
clfANN = MLPClassifier(solver='adam', activation='relu',
                    batch_size=10, tol=1e-5,
                     hidden_layer_sizes=(50,), random_state=1, max_iter=100000, verbose=50)
 
#train the classifiers                       
clfDT.fit(x_train, y_train)
clfSVMr.fit(x_train, y_train)
clfSVMp.fit(x_train, y_train)
clfNB.fit(x_train, y_train)
clfNN.fit(x_train, y_train)
clfRF.fit(x_train, y_train)
clfRFmd.fit(x_train, y_train)
clfADA.fit(x_train, y_train)
clfANN.fit(x_train, y_train)


###feature importance
fi_DT=clfDT.fit(x_train, y_train).feature_importances_
fi_DT
indicesDT = np.argsort(fi_DT)[::-1]
indicesDT

fi_RF=clfRF.fit(x_train, y_train).feature_importances_
fi_RF
indicesRF = np.argsort(fi_RF)[::-1]
indicesRF

fi_RFmd=clfRFmd.fit(x_train, y_train).feature_importances_
fi_RFmd
indicesRFmd = np.argsort(fi_RFmd)[::-1]
indicesRFmd

fi_clfADA=clfADA.fit(x_train, y_train).feature_importances_
fi_clfADA
indicesclfADA = np.argsort(fi_clfADA)[::-1]
indicesclfADA





#test the trained model on the test set
y_test_pred_DT=clfDT.predict(x_test)
y_test_pred_SVMr=clfSVMr.predict(x_test)
y_test_pred_SVMp=clfSVMp.predict(x_test)
y_test_pred_NB=clfNB.predict(x_test)
y_test_pred_NN=clfNN.predict(x_test)
y_test_pred_RF=clfRF.predict(x_test)
y_test_pred_RFmd=clfRFmd.predict(x_test)
y_test_pred_ADA=clfADA.predict(x_test)
y_test_pred_ANN=clfANN.predict(x_test)


# Measures of performance: Precision, Recall, F1
print ('DecTree: Macro Precision, recall, f1-score')
print ( precision_recall_fscore_support(y_test, y_test_pred_DT, average='macro'))
print ('DecTree: Micro Precision, recall, f1-score')
print (precision_recall_fscore_support(y_test, y_test_pred_DT, average='micro'))
print ('\n')

print ('NearNeigh: Macro Precision, recall, f1-score')
print ( precision_recall_fscore_support(y_test, y_test_pred_NN, average='macro'))
print ('NearNeigh: Micro Precision, recall, f1-score')
print (precision_recall_fscore_support(y_test, y_test_pred_NN, average='micro'))
print ('\n')


clfSVMr= svm.SVC(C=100.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovo', degree=1, gamma='auto', kernel='rbf',
    max_iter=-1, random_state=None, shrinking=True,
    tol=0.001, verbose=False, probability=True)

clfSVMr.fit(x_train, y_train)
y_test_pred_SVMr=clfSVMr.predict(x_test)

print ('Support Vector RBF: Macro Precision, recall, f1-score')
print ( precision_recall_fscore_support(y_test, y_test_pred_SVMr, average='macro'))
print ('Support Vector RBF: Micro Precision, recall, f1-score')
print (precision_recall_fscore_support(y_test, y_test_pred_SVMr, average='micro'))
print ('\n')

clfSVMp= svm.SVC(C=100.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovo', degree=1, gamma='auto', kernel='poly',
    max_iter=-1, random_state=None, shrinking=True,
    tol=0.001, verbose=False, probability=True)

clfSVMp.fit(x_train, y_train)
y_test_pred_SVMp=clfSVMp.predict(x_test)

print ('Support Vector Poly: Macro Precision, recall, f1-score')
print ( precision_recall_fscore_support(y_test, y_test_pred_SVMp, average='macro'))
print ('Support Vector Poly: Micro Precision, recall, f1-score')
print (precision_recall_fscore_support(y_test, y_test_pred_SVMp, average='micro'))
print ('\n')

#Define a Naive Bayes
clfNB = GaussianNB()

clfNB.fit(x_train, y_train)
y_test_pred_NB=clfNB.predict(x_test)

print ('Naive Bayes: Macro Precision, recall, f1-score')
print ( precision_recall_fscore_support(y_test, y_test_pred_NB, average='macro'))
print ('Naive Bayes: Micro Precision, recall, f1-score')
print (precision_recall_fscore_support(y_test, y_test_pred_NB, average='micro'))
print ('\n')


#random Forest
clfRF = RandomForestClassifier(n_estimators=5, max_depth=5)
clfRF.fit(x_train, y_train)
y_test_pred_RF=clfRF.predict(x_test)

print ('Random Forest: Macro Precision, recall, f1-score')
print ( precision_recall_fscore_support(y_test, y_test_pred_RF, average='macro'))
print ('Random Forest: Micro Precision, recall, f1-score')
print (precision_recall_fscore_support(y_test, y_test_pred_RF, average='micro'))
print ('\n')


clfRFmd = RandomForestClassifier(n_estimators=2, max_depth=2, random_state=None)
clfRFmd.fit(x_train, y_train) 
y_test_pred_RFmd=clfRFmd.predict(x_test)

print ('Random Forest MaxDepth: Macro Precision, recall, f1-score')
print ( precision_recall_fscore_support(y_test, y_test_pred_RFmd, average='macro'))
print ('Random Forest MaxDepth: Micro Precision, recall, f1-score')
print (precision_recall_fscore_support(y_test, y_test_pred_RFmd, average='micro'))
print ('\n')


#ADA Boost
clfADA = AdaBoostClassifier(n_estimators=8, learning_rate=1)
clfADA.fit(x_train, y_train)
y_test_pred_ADA=clfADA.predict(x_test)

print ('ADA Boost: Macro Precision, recall, f1-score')
print ( precision_recall_fscore_support(y_test, y_test_pred_ADA, average='macro'))
print ('ADA Boost: Micro Precision, recall, f1-score')
print (precision_recall_fscore_support(y_test, y_test_pred_ADA, average='micro'))
print ('\n')

#ANN Classifier
clfANN = MLPClassifier(solver='adam', activation='relu',
                    batch_size=10, tol=1e-5,
                     hidden_layer_sizes=(50,), random_state=1, max_iter=100000, verbose=50)

clfANN.fit(x_train, y_train)
y_test_pred_ANN=clfANN.predict(x_test)

print ('ANN: Macro Precision, recall, f1-score')
print ( precision_recall_fscore_support(y_test, y_test_pred_ANN, average='macro'))
print ('ANN: Micro Precision, recall, f1-score')
print (precision_recall_fscore_support(y_test, y_test_pred_ANN, average='micro'))
print ('\n')



pr_y_test_pred_DT=clfDT.predict_proba(x_test)
#pr_y_test_pred_SVMr=clfSVMr.predict_proba(x_test)
#pr_y_test_pred_SVMp=clfSVMp.predict_proba(x_test)
pr_y_test_pred_NN=clfNN.predict_proba(x_test)
pr_y_test_pred_NB=clfNB.predict_proba(x_test)
pr_y_test_pred_RF=clfRF.predict_proba(x_test)
pr_y_test_pred_RFmd=clfRFmd.predict_proba(x_test)
pr_y_test_pred_ADA=clfADA.predict_proba(x_test)
pr_y_test_pred_ANN=clfANN.predict_proba(x_test)

#clfSVMr.predict_proba
#clfSVMp.predict_proba

#ROC curve
fprDT, tprDT, thresholdsDT = roc_curve(y_test, pr_y_test_pred_DT[:,1])
#fprSVMr, tprSVMr, thresholdsSVMr = roc_curve(y_test, pr_y_test_pred_SVMr[:,1])
#fprSVMp, tprSVMp, thresholdsSVMp = roc_curve(y_test, pr_y_test_pred_SVMp[:,1])
fprNN, tprNN, thresholdsNN = roc_curve(y_test, pr_y_test_pred_NN[:,1])
fprNB, tprNB, thresholdsNB = roc_curve(y_test, pr_y_test_pred_NB[:,1])
fprRF, tprRF, thresholdsRF = roc_curve(y_test, pr_y_test_pred_RF[:,1])
fprRFmd, tprRFmd, thresholdsRFmd = roc_curve(y_test, pr_y_test_pred_RFmd[:,1])
fprADA, tprADA, thresholdsADA = roc_curve(y_test, pr_y_test_pred_ADA[:,1])
fprANN, tprANN, thresholdsANN = roc_curve(y_test, pr_y_test_pred_ANN[:,1])

lw=2
plt.plot(fprDT,tprDT,color='brown',label='Decision Tree')
#plt.plot(fprSVMr,tprSVMr,color='red',label='Support Vector RBF')
#plt.plot(fprSVMp,tprSVMp,color='purple',label='Support Vector Poly')
plt.plot(fprNN,tprNN,color='blue',label='Nearest Neighbor')
plt.plot(fprNB,tprNB,color='green',label='Naive Bayes')
plt.plot(fprRF,tprRF,color='orange',label='Random Forest')
plt.plot(fprRFmd,tprRFmd,color='pink',label='Random Forest MaxDepth')
plt.plot(fprADA,tprADA,color='navy',label='ADA Boost')
plt.plot(fprANN,tprANN,color='yellow',label='ANN')

plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


#####k means

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs

#generate some random data
#X, y_true = make_blobs(n_samples=300, centers=7,
    ##                   cluster_std=0.60, random_state=0)

#plot them
#plt.figure(1)
#plt.scatter(X[:, 0], X[:, 1], s=50)


#run clusterings for differen values of k
inertiasAll=[]
silhouettesAll=[]
for n in range(2,12):
    print ('Clustering for n=',n)
    kmeans = KMeans(n_clusters=n)
    kmeans.fit(x)
    y_kmeans = kmeans.predict(x)

#get cluster centers
    kmeans.cluster_centers_

#evalute
    print ('inertia=',kmeans.inertia_)
    silhouette_values = silhouette_samples(x, y_kmeans)
    print ('silhouette=', np.mean(silhouette_values))
    
    inertiasAll.append(kmeans.inertia_)
    silhouettesAll.append(np.mean(silhouette_values))    


# plot data acoding to the cluster they belong
plt.figure(2)
plt.scatter(x[:, 0], x[:, 1], c=y_kmeans, s=20, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);

plt.figure(3)
plt.plot(range(2,12),silhouettesAll,'r*-')
plt.ylabel('Silhouette score')
plt.xlabel('Number of clusters')
plt.figure(4)
plt.plot(range(2,12),inertiasAll,'g*-')
plt.ylabel('Inertia Score')
plt.xlabel('Number of clusters')

####K fold


from sklearn.model_selection import KFold,cross_val_score

kf = KFold(n_splits=10)
kf.get_n_splits(x)

#print(kf)  
#KFold(n_splits=10, random_state=None, shuffle=False)
 #for train_index, test_index in kf.split(x):
   # print("TRAIN:", train_index, "TEST:", test_index)
  #  x_train, x_test = x[train_index], x[test_index]
    #y_train, y_test = y[train_index], y[test_index]

#kfold decision tree

clf_tree=tree.DecisionTreeClassifier()

#score_array =[]
for train_index, test_index in kf.split(x):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf=clf_tree.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
   # score_array.append(precision_recall_fscore_support(y_test, y_pred, average=None))

#avg_score = np.mean(score_array,axis=0)
#print(avg_score)


#kf = KFold(n_splits=10)

#clf_tree=tree.DecisionTreeClassifier()
scores = cross_val_score(clf, x, y, cv=kf)
print(scores)

#avg_score = np.mean(score_array)
#print(avg_score)

clf_nb=clfnb = GaussianNB()

#score_array =[]
for train_index, test_index in kf.split(x):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clfnb=clf_nb.fit(x_train,y_train)
    y_pred = clfnb.predict(x_test)
   # score_array.append(precision_recall_fscore_support(y_test, y_pred, average=None))

#avg_score = np.mean(score_array,axis=0)
#print(avg_score)


#kf = KFold(n_splits=10)

#clf_tree=tree.DecisionTreeClassifier()
scores = cross_val_score(clfnb, x, y, cv=kf)
print(scores)



