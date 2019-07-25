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
#from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
#import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from sklearn.metrics import accuracy_score



data = pd.read_csv('C:\\Users\\Rachel\\Documents\\Capstone Project\\final8_56.csv', header=0)

data.info()
data.head()
data.columns

#determining which columns are part of data and which is the prediction. Removed the user_id and session_id.
x = data.loc[:,'mobile':'duration_sec']
y = data.loc[:,'click_out']

#split data sets into training and testing
x=preprocessing.scale(x)
test_size = 0.5
x_train, x_test, y_train, y_test = train_test_split (x,y, test_size=0.5, random_state=1)


#Define decision Tree
clfDT =  tree.DecisionTreeClassifier()
clfDT.fit(x_train, y_train)
y_test_pred_DT=clfDT.predict(x_test)
print ('DecTree: Macro Precision, recall, f1-score')
print ( precision_recall_fscore_support(y_test, y_test_pred_DT, average='macro'))
print ('DecTree: Micro Precision, recall, f1-score')
print (precision_recall_fscore_support(y_test, y_test_pred_DT, average='micro'))
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

#Define a Nearest Neighbours classifiers 
#n_neighbors: is the number of neighbors
#metric: is the distance measure,
clfNN = KNeighborsClassifier(n_neighbors=3)
clfNN.fit(x_train, y_train)
y_test_pred_NN=clfNN.predict(x_test)
print ('NearNeigh: Macro Precision, recall, f1-score')
print ( precision_recall_fscore_support(y_test, y_test_pred_NN, average='macro'))
print ('NearNeigh: Micro Precision, recall, f1-score')
print (precision_recall_fscore_support(y_test, y_test_pred_NN, average='micro'))
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

#random Forest2
clfRF2 = RandomForestClassifier(n_estimators=5, max_depth=100)
clfRF2.fit(x_train, y_train)
y_test_pred_RF2=clfRF2.predict(x_test)

print ('Random Forest2: Macro Precision, recall, f1-score')
print ( precision_recall_fscore_support(y_test, y_test_pred_RF2, average='macro'))
print ('Random Forest2: Micro Precision, recall, f1-score')
print (precision_recall_fscore_support(y_test, y_test_pred_RF2, average='micro'))
print ('\n')

#random Forest 3
clfRF3 = RandomForestClassifier(n_estimators=50, max_depth=100)
clfRF3.fit(x_train, y_train)
y_test_pred_RF3=clfRF3.predict(x_test)

print ('Random Forest3: Macro Precision, recall, f1-score')
print ( precision_recall_fscore_support(y_test, y_test_pred_RF3, average='macro'))
print ('Random Forest3: Micro Precision, recall, f1-score')
print (precision_recall_fscore_support(y_test, y_test_pred_RF3, average='micro'))
print ('\n')

#random Forest 4
clfRF4 = RandomForestClassifier(n_estimators=200, max_depth=100)
clfRF4.fit(x_train, y_train)
y_test_pred_RF4=clfRF4.predict(x_test)

print ('Random Forest4: Macro Precision, recall, f1-score')
print ( precision_recall_fscore_support(y_test, y_test_pred_RF4, average='macro'))
print ('Random Forest4: Micro Precision, recall, f1-score')
print (precision_recall_fscore_support(y_test, y_test_pred_RF4, average='micro'))
print ('\n')

#random Forest 5
clfRF5 = RandomForestClassifier(n_estimators=400, max_depth=100)
clfRF5.fit(x_train, y_train)
y_test_pred_RF5=clfRF5.predict(x_test)

print ('Random Forest5: Macro Precision, recall, f1-score')
print ( precision_recall_fscore_support(y_test, y_test_pred_RF5, average='macro'))
print ('Random Forest5: Micro Precision, recall, f1-score')
print (precision_recall_fscore_support(y_test, y_test_pred_RF5, average='micro'))
print ('\n')

#random Forest 6
clfRF6 = RandomForestClassifier(n_estimators=400, max_depth=200)
clfRF6.fit(x_train, y_train)
y_test_pred_RF6=clfRF6.predict(x_test)

print ('Random Forest6: Macro Precision, recall, f1-score')
print ( precision_recall_fscore_support(y_test, y_test_pred_RF6, average='macro'))
print ('Random Forest6: Micro Precision, recall, f1-score')
print (precision_recall_fscore_support(y_test, y_test_pred_RF6, average='micro'))
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


#ANN2 Classifier - parameters (50,100)
clfANN2 = MLPClassifier(solver='adam', activation='relu',
                    batch_size=10, tol=1e-5,
                     hidden_layer_sizes=(50,100), random_state=1, max_iter=100000, verbose=50)

clfANN2.fit(x_train, y_train)
y_test_pred_ANN2=clfANN2.predict(x_test)

print ('ANN2: Macro Precision, recall, f1-score')
print ( precision_recall_fscore_support(y_test, y_test_pred_ANN2, average='macro'))
print ('ANN2: Micro Precision, recall, f1-score')
print (precision_recall_fscore_support(y_test, y_test_pred_ANN2, average='micro'))
print ('\n')

#ANN3 Classifier - parameters (100,100)
clfANN3 = MLPClassifier(solver='adam', activation='relu',
                    batch_size=10, tol=1e-5,
                     hidden_layer_sizes=(100,100), random_state=1, max_iter=100000, verbose=50)

clfANN3.fit(x_train, y_train)
y_test_pred_ANN3=clfANN3.predict(x_test)

print ('ANN3: Macro Precision, recall, f1-score')
print ( precision_recall_fscore_support(y_test, y_test_pred_ANN3, average='macro'))
print ('ANN3: Micro Precision, recall, f1-score')
print (precision_recall_fscore_support(y_test, y_test_pred_ANN3, average='micro'))
print ('\n')

#ANN4 Classifier - parameters (100,100,50)
clfANN4 = MLPClassifier(solver='adam', activation='relu',
                    batch_size=10, tol=1e-5,
                     hidden_layer_sizes=(100,100,50), random_state=1, max_iter=100000, verbose=50)

clfANN4.fit(x_train, y_train)
y_test_pred_ANN4=clfANN4.predict(x_test)

print ('ANN4: Macro Precision, recall, f1-score')
print ( precision_recall_fscore_support(y_test, y_test_pred_ANN4, average='macro'))
print ('ANN4: Micro Precision, recall, f1-score')
print (precision_recall_fscore_support(y_test, y_test_pred_ANN4, average='micro'))
print ('\n')



pr_y_test_pred_DT=clfDT.predict_proba(x_test)
pr_y_test_pred_NN=clfNN.predict_proba(x_test)
pr_y_test_pred_NB=clfNB.predict_proba(x_test)
pr_y_test_pred_RF6=clfRF6.predict_proba(x_test)
pr_y_test_pred_RFmd=clfRFmd.predict_proba(x_test)
pr_y_test_pred_ADA=clfADA.predict_proba(x_test)
pr_y_test_pred_ANN3=clfANN3.predict_proba(x_test)


#ROC curve
fprDT, tprDT, thresholdsDT = roc_curve(y_test, pr_y_test_pred_DT[:,1])
fprNN, tprNN, thresholdsNN = roc_curve(y_test, pr_y_test_pred_NN[:,1])
fprNB, tprNB, thresholdsNB = roc_curve(y_test, pr_y_test_pred_NB[:,1])
fprRF, tprRF, thresholdsRF = roc_curve(y_test, pr_y_test_pred_RF6[:,1])
fprRFmd, tprRFmd, thresholdsRFmd = roc_curve(y_test, pr_y_test_pred_RFmd[:,1])
fprADA, tprADA, thresholdsADA = roc_curve(y_test, pr_y_test_pred_ADA[:,1])
fprANN, tprANN, thresholdsANN = roc_curve(y_test, pr_y_test_pred_ANN3[:,1])

lw=2
plt.plot(fprDT,tprDT,color='red',label='Decision Tree')
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
plt.title('Receiver Operating Characteristic Original Data')
plt.legend(loc="lower right")
plt.show()



#PCA
sc = StandardScaler()
x = sc.fit_transform(x)

pca = PCA()
pca.fit(x)

# Percentage of variance explained for each components
print('explained variance ratio for each component: %s'
      % str(pca.explained_variance_ratio_.round(2)))

fig = plt.figure(1,figsize=(12,6))
fig.add_subplot(1,2,1)
plt.bar(np.arange(pca.n_components_), 100*pca.explained_variance_ratio_)
plt.title('Relative information content of PCA components')
plt.xlabel("PCA component number")
plt.ylabel("PCA component variance % ")

plt.figure(2)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');

# So I will keep now only the 50 principal components when i create the PCA object
pca1 = PCA(n_components=50)
pca1.fit(x)

x_pca = pca1.transform(x)

x.shape
x_pca.shape

x_pca_train, x_pca_test, y_pca_train, y_pca_test = train_test_split(x_pca, y, test_size=0.5, random_state=1, shuffle=True)

#Rerun best performing classifiers

#PCA decision Tree
clfDTPCA =  tree.DecisionTreeClassifier()
clfDTPCA.fit(x_pca_train, y_pca_train)
y_test_pred_DTPCA = clfDTPCA.predict(x_pca_test)
print ('DecTreePCA: Macro Precision, recall, f1-score')
print ( precision_recall_fscore_support(y_pca_test, y_test_pred_DTPCA, average='macro'))
print ('DecTreePCA: Micro Precision, recall, f1-score')
print (precision_recall_fscore_support(y_pca_test, y_test_pred_DTPCA, average='micro'))
print ('\n')

#PCA Nearest Neighbours classifiers 
clfNNPCA = KNeighborsClassifier(n_neighbors=3)
clfNNPCA.fit(x_pca_train, y_pca_train)
y_test_pred_NNPCA = clfNNPCA.predict(x_pca_test)
print ('NearNeighPCA: Macro Precision, recall, f1-score')
print ( precision_recall_fscore_support(y_pca_test, y_test_pred_NNPCA, average='macro'))
print ('NearNeighPCA: Micro Precision, recall, f1-score')
print (precision_recall_fscore_support(y_pca_test, y_test_pred_NNPCA, average='micro'))
print ('\n') 

#PCA random Forest 6
clfRF6PCA = RandomForestClassifier(n_estimators=400, max_depth=200)
clfRF6PCA.fit(x_pca_train, y_pca_train)
y_test_pred_RF6PCA = clfRF6PCA.predict(x_pca_test)

print ('Random Forest6PCA: Macro Precision, recall, f1-score')
print ( precision_recall_fscore_support(y_pca_test, y_test_pred_RF6PCA, average='macro'))
print ('Random Forest6PCA: Micro Precision, recall, f1-score')
print (precision_recall_fscore_support(y_pca_test, y_test_pred_RF6PCA, average='micro'))
print ('\n')

# PCA ADA Boost
clfADAPCA = AdaBoostClassifier(n_estimators=8, learning_rate=1)
clfADAPCA.fit(x_pca_train, y_pca_train)
y_test_pred_ADAPCA = clfADAPCA.predict(x_pca_test)

print ('ADA BoostPCA: Macro Precision, recall, f1-score')
print ( precision_recall_fscore_support(y_pca_test, y_test_pred_ADAPCA, average='macro'))
print ('ADA BoostPCA: Micro Precision, recall, f1-score')
print (precision_recall_fscore_support(y_pca_test, y_test_pred_ADAPCA, average='micro'))
print ('\n')

#PCA ANN3 Classifier - parameters (100,100)
clfANN3PCA = MLPClassifier(solver='adam', activation='relu',
                    batch_size=10, tol=1e-5,
                     hidden_layer_sizes=(100,100), random_state=1, max_iter=100000, verbose=50)

clfANN3PCA.fit(x_pca_train, y_pca_train)
y_test_pred_ANN3PCA = clfANN3PCA.predict(x_pca_test)

print ('ANN3PCA: Macro Precision, recall, f1-score')
print ( precision_recall_fscore_support(y_pca_test, y_test_pred_ANN3PCA, average='macro'))
print ('ANN3PCA: Micro Precision, recall, f1-score')
print (precision_recall_fscore_support(y_pca_test, y_test_pred_ANN3PCA, average='micro'))
print ('\n')

pr_y_test_pred_DTPCA=clfDTPCA.predict_proba(x_pca_test)
pr_y_test_pred_NNPCA=clfNNPCA.predict_proba(x_pca_test)
pr_y_test_pred_RF6PCA=clfRF6PCA.predict_proba(x_pca_test)
pr_y_test_pred_ADAPCA=clfADAPCA.predict_proba(x_pca_test)
pr_y_test_pred_ANN3PCA=clfANN3PCA.predict_proba(x_pca_test)


#ROC curve PCA
fprDTPCA, tprDTPCA, thresholdsDTPCA = roc_curve(y_pca_test, pr_y_test_pred_DTPCA[:,1])
fprNNPCA, tprNNPCA, thresholdsNNPCA = roc_curve(y_pca_test, pr_y_test_pred_NNPCA[:,1])
fprRFPCA, tprRFPCA, thresholdsRFPCA = roc_curve(y_pca_test, pr_y_test_pred_RF6PCA[:,1])
fprADAPCA, tprADAPCA, thresholdsADAPCA = roc_curve(y_pca_test, pr_y_test_pred_ADAPCA[:,1])
fprANNPCA, tprANNPCA, thresholdsANNPCA = roc_curve(y_pca_test, pr_y_test_pred_ANN3PCA[:,1])

lw=2
plt.plot(fprDTPCA,tprDTPCA,color='red',label='Decision Tree')
plt.plot(fprNNPCA,tprNNPCA,color='blue',label='Nearest Neighbor')
plt.plot(fprRFPCA,tprRFPCA,color='orange',label='Random Forest')
plt.plot(fprADAPCA,tprADAPCA,color='navy',label='ADA Boost')
plt.plot(fprANNPCA,tprANNPCA,color='yellow',label='ANN')

plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic PCA Original Data')
plt.legend(loc="lower right")
plt.show()



