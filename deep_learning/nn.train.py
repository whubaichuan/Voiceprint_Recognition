# ==============================================================
#   Copyright (C) 2021 whubaichuan. All rights reserved.
#   function： Demo of voiceprint recognition among three people by a vanilla CNN method
#              This file is the training step.
# ==============================================================
#   Create by whubaichuan at 2021.05.01
#   Version 1.0
#   whubaichuan [huangbaichuan@whu.edu.cn]
# ==============================================================

from CNN_net import SimpleVGGNet
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import utils_paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os

path="C:/Users/work_computer/Desktop/Voiceprint_Recognition/deep_learning/"
image_width = 64

# load data
print("------Begin load data------")
data = []
labels = []

baichuan_plots = [] 
for (_,_,filenames) in os.walk( path+"train_data/baichuan/" ): 
    baichuan_plots.extend(filenames) 
    break 

for baichuan_plot in baichuan_plots: 
    image = cv2.imread( path+"train_data/baichuan/" + baichuan_plot)
    image = cv2.resize(image,(image_width,image_width))
    data.append(image) 
    labels.append(0) 

# shiyer_plots = [] 
# for (_,_,filenames) in os.walk(path+"train_data/shiyer/" ): 
#     shiyer_plots.extend(filenames) 
#     break 

# for shiyer_plot in shiyer_plots: 
#     image = cv2.imread( path+"train_data/shiyer/" + shiyer_plot)
#     image = cv2.resize(image,(image_width,image_width))
#     data.append(image) 
#     labels.append(1) 


# zhou_plots = [] 
# for (_,_,filenames) in os.walk(path+"train_data/zhou/" ): 
#     zhou_plots.extend(filenames) 
#     break 

# for zhou_plot in zhou_plots: 
#     image = cv2.imread( path+"train_data/zhou/" + zhou_plot)
#     image = cv2.resize(image,(image_width,image_width))
#     data.append(image) 
#     labels.append(1) 


wu_plots = [] 
for (_,_,filenames) in os.walk(path+"train_data/wu/" ): 
    wu_plots.extend(filenames) 
    break 

for wu_plot in wu_plots: 
    image = cv2.imread( path+"train_data/wu/" + wu_plot)
    image = cv2.resize(image,(image_width,image_width))
    data.append(image) 
    labels.append(1) 


tang_plots = [] 
for (_,_,filenames) in os.walk(path+"train_data/tang/" ): 
    tang_plots.extend(filenames) 
    break 

for tang_plot in tang_plots: 
    image = cv2.imread( path+"train_data/tang/" + tang_plot)
    image = cv2.resize(image,(image_width,image_width))
    data.append(image) 
    labels.append(2) 


# normalization
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)


(trainX, testX, trainY, testY) = train_test_split(data,labels, test_size=0.3, shuffle=True,random_state=42)

#one-hot encoding
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# data
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
    horizontal_flip=True, fill_mode="nearest")

model = SimpleVGGNet.build(width=image_width, height=image_width, depth=3,classes=len(lb.classes_))

# initial_parameter
INIT_LR = 0.01
EPOCHS = 200
BS = 8

print("------Start training the VGG model------")
#opt = SGD(lr=INIT_LR, decay=INIT_LR / EPOCHS)
opt = Adam(lr=INIT_LR,beta_1=0.9,beta_2=0.99,epsilon=1e-8,decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])

# H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
#     validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
#     epochs=EPOCHS)

H = model.fit(trainX, trainY, validation_data=(testX, testY),
    epochs=EPOCHS, batch_size=BS)


# validation
print("------Validation step------")
predictions = model.predict(testX, batch_size=4)
print (accuracy_score(testY.argmax(axis=1), predictions.argmax(axis=1))) 
print(f1_score(testY.argmax(axis=1), predictions.argmax(axis=1), labels=None, pos_label=1, average='micro',sample_weight=None))
print(confusion_matrix(testY.argmax(axis=1),predictions.argmax(axis=1)))

# plot the results
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.xlim(0,200)
plt.ylim(0,5)
plt.plot(N, H.history["loss"], label="train_loss")
#plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["accuracy"], label="train_acc")
plt.plot(N, H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(path+"output/cnn_plot.png")

print("------save the model------")
model.save(path+"/output/cnn.model")
f = open(path+"output/cnn_lb.pickle", "wb")
f.write(pickle.dumps(lb))
f.close()