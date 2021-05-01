# ==============================================================
#   Copyright (C) 2021 whubaichuan. All rights reserved.
#   function： Demo of voiceprint recognition among three people by SVM method
# ==============================================================
#   Create by whubaichuan at 2021.05.01
#   Version 1.0
#   whubaichuan [huangbaichuan@whu.edu.cn]
# ==============================================================


import os 
import librosa
import librosa.feature
import librosa.display
from sklearn.model_selection import train_test_split 
from sklearn.svm import LinearSVC 
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np 

path="C:/Users/work_computer/Desktop/NTNU/machine_learning/"

#extract the MFCC features
def get_features(song): 
    y,_ = librosa.load(song)
    mfcc = librosa.feature.mfcc(y)
    mfcc /= np.amax(np.absolute(mfcc))
    return np.ndarray.flatten(mfcc)

#=============construct the training dataset (choose three of five)=============
X = [] 
y = [] 

baichuan_plots = [] 
for (_,_,filenames) in os.walk( path+"train_data/baichuan/" ): 
    baichuan_plots.extend(filenames) 
    break 

for baichuan_plot in baichuan_plots: 
    X.append(get_features( path+"train_data/baichuan/" + baichuan_plot)) 
    y.append(0) 

# shiyer_plots = [] 
# for (_,_,filenames) in os.walk(path+"train_data/shiyer/" ): 
#     shiyer_plots.extend(filenames) 
#     break 

# for shiyer_plot in shiyer_plots: 
#     X.append(get_features( path+"train_data/shiyer/" + shiyer_plot)) 
#     y.append(1) 

zhou_plots = [] 
for (_,_,filenames) in os.walk(path+"train_data/zhou/" ): 
    zhou_plots.extend(filenames) 
    break 

for zhou_plot in zhou_plots: 
    X.append(get_features( path+"train_data/zhou/" + zhou_plot)) 
    y.append(1) 


# wu_plots = [] 
# for (_,_,filenames) in os.walk(path+"train_data/wu/" ): 
#     wu_plots.extend(filenames) 
#     break 

# for wu_plot in wu_plots: 
#     X.append(get_features( path+"train_data/wu/" + wu_plot)) 
#     y.append(1) 

tang_plots = [] 
for (_,_,filenames) in os.walk(path+"train_data/tang/" ): 
    tang_plots.extend(filenames) 
    break 

for tang_plot in tang_plots: 
    X.append(get_features( path+"train_data/tang/" + tang_plot)) 
    y.append(2) 
#========================================================================



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y) 

#training
clf = LinearSVC(random_state=0, tol=1e-5) 
clf.fit(X_train, y_train) 
predicted = clf.predict(X_test) 

#get the training result
print (accuracy_score(y_test, predicted)) 
print(f1_score(y_test, predicted, labels=None, pos_label=1, average='micro',sample_weight=None))
print(confusion_matrix(y_test, predicted))
target_names = ['baichuan', 'shiyer','tang']
print(classification_report(y_test, predicted, target_names=target_names))


#=========construct the testing dataset (choose three of five)============
X_predict=[]
y_predict=[]

baichuan_plots = [] 
for (_,_,filenames) in os.walk( path+"test_data/baichuan_test/" ): 
    baichuan_plots.extend(filenames) 
    break 

for baichuan_plot in baichuan_plots: 
    X_predict.append(get_features( path+"test_data/baichuan_test/" + baichuan_plot)) 
    y_predict.append(0) 

# shiyer_plots = [] 
# for (_,_,filenames) in os.walk( path+"test_data/shiyer_test/" ): 
#     shiyer_plots.extend(filenames) 
#     break 

# for shiyer_plot in shiyer_plots: 
#     X_predict.append(get_features( path+"test_data/shiyer_test/" + shiyer_plot)) 
#     y_predict.append(1) 


zhou_plots = [] 
for (_,_,filenames) in os.walk( path+"test_data/zhou_test/" ): 
    zhou_plots.extend(filenames) 
    break 

for zhou_plot in zhou_plots: 
    X_predict.append(get_features( path+"test_data/zhou_test/" + zhou_plot)) 
    y_predict.append(1) 


# wu_plots = [] 
# for (_,_,filenames) in os.walk( path+"test_data/wu_test/" ): 
#     wu_plots.extend(filenames) 
#     break 

# for wu_plot in wu_plots: 
#     X_predict.append(get_features( path+"test_data/wu_test/" + wu_plot)) 
#     y_predict.append(1) 

tang_plots = [] 
for (_,_,filenames) in os.walk(path+"test_data/tang_test/" ): 
    tang_plots.extend(filenames) 
    break 

for tang_plot in tang_plots: 
    X_predict.append(get_features( path+"test_data/tang_test/" + tang_plot)) 
    y_predict.append(2) 
#==============================================================


#get the testing result
predicted_predict = clf.predict(X_predict) 
print (accuracy_score(y_predict, predicted_predict)) 
print(f1_score(y_predict, predicted_predict, labels=None, pos_label=1, average='micro', sample_weight=None))
print(confusion_matrix(y_predict, predicted_predict))
print(classification_report(y_predict, predicted_predict, target_names=target_names))
