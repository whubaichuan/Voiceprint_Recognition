# ==============================================================
#   Copyright (C) 2021 whubaichuan. All rights reserved.
#   function： Demo of voiceprint recognition among three people by a vanilla CNN method
#              This file is the testing step.
# ==============================================================
#   Create by whubaichuan at 2021.05.01
#   Version 1.0
#   whubaichuan [huangbaichuan@whu.edu.cn]
# ==============================================================

from keras.models import load_model
import argparse
import pickle
import cv2
import os
import numpy as np

path="C:/Users/work_computer/Desktop/Voiceprint_Recognition/deep_learning/"
image_width = 64

# ===========================load testing data===========================
data = []
labels = []

baichuan_plots = [] 
for (_,_,filenames) in os.walk( path+"test_data/baichuan/" ): 
    baichuan_plots.extend(filenames) 
    break 

for baichuan_plot in baichuan_plots: 
    image = cv2.imread( path+"test_data/baichuan/" + baichuan_plot)
    image = cv2.resize(image,(image_width,image_width))
    data.append(image) 
    labels.append(0) 

# shiyer_plots = [] 
# for (_,_,filenames) in os.walk(path+"test_data/shiyer/" ): 
#     shiyer_plots.extend(filenames) 
#     break 

# for shiyer_plot in shiyer_plots: 
#     image = cv2.imread( path+"test_data/shiyer/" + shiyer_plot)
#     image = cv2.resize(image,(image_width,image_width))
#     data.append(image) 
#     labels.append(1) 


# zhou_plots = [] 
# for (_,_,filenames) in os.walk(path+"test_data/zhou/" ): 
#     zhou_plots.extend(filenames) 
#     break 

# for zhou_plot in zhou_plots: 

#     image = cv2.imread(path+"test_data/zhou/" + zhou_plot)
#     image = cv2.resize(image,(image_width,image_width))
#     data.append(image) 
#     labels.append(1) 

wu_plots = [] 
for (_,_,filenames) in os.walk(path+"test_data/wu/" ): 
    wu_plots.extend(filenames) 
    break 

for wu_plot in wu_plots: 
    image = cv2.imread( path+"test_data/wu/" + wu_plot)
    image = cv2.resize(image,(image_width,image_width))
    data.append(image) 
    labels.append(1) 


tang_plots = [] 
for (_,_,filenames) in os.walk(path+"test_data/tang/" ): 
    tang_plots.extend(filenames) 
    break 

for tang_plot in tang_plots: 
    image = cv2.imread( path+"test_data/tang/" + tang_plot)
    image = cv2.resize(image,(image_width,image_width))
    data.append(image) 
    labels.append(2) 
# ==============================================================

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)


print("------load model and label------")
model = load_model(path+"output/cnn.model")
lb = pickle.loads(open(path+"output/cnn_lb.pickle", "rb").read())

predict_lable=[]
# predict
for data_item in data:
    data_item = data_item.reshape((1, data_item.shape[0], data_item.shape[1],data_item.shape[2]))
    preds = model.predict(data_item)

    # get the prediction and the corresponding label
    i = preds.argmax(axis=1)[0]
    label = lb.classes_[i]
    predict_lable.append(label)

# calculate the accuracy
count_right = 0
i = 0 
while i<15:
    if i <5:
        if predict_lable[i]==0:
            count_right+=1
    if i>4 and i<10:
        if predict_lable[i]==1:
            count_right+=1
    if i>9:
        if predict_lable[i]==2:
            count_right+=1
    i +=1
print(count_right/15)
