# ==============================================================
#   Copyright (C) 2021 whubaichuan. All rights reserved.
#   function： Preprocessing steps of .wav file.
# ==============================================================
#   Create by whubaichuan at 2021.05.01
#   Version 1.0
#   whubaichuan [huangbaichuan@whu.edu.cn]
# ==============================================================

#%%
#split the .wav into each 12s fragment
from pydub import AudioSegment 
import os
count=1 
for i in range(1,720,12): 
    t1 = i * 1000 #Works in milliseconds 
    t2 = (i+12) * 1000 
    newAudio = AudioSegment.from_wav( "C:/Users/work_computer/Desktop/NTNU/wu.wav" ) 
    newAudio = newAudio[t1:t2] 
    newAudio.export( "C:/Users/work_computer/Desktop/NTNU/wu/" +str(count)+ '.wav' , format= "wav" ) #Exports to a wav file in the current path. 
    print(count) 
    count+=1 

#%%
#plot the wav of sound
from scipy.io.wavfile import read 
import matplotlib.pyplot as plt 
from os import walk 

baichuan_wavs = [] 
for (_,_,filenames) in walk( "C:/Users/work_computer/Desktop/NTNU/machine_learning/shiyer_test_all/" ): 
    baichuan_wavs.extend(filenames) 
    break 
for baichuan_wav in baichuan_wavs:  
    input_data = read("C:/Users/work_computer/Desktop/NTNU/machine_learning/shiyer_test_all/" + baichuan_wav)
    audio = input_data[1] 
    # plot the first 1024 samples 
    plt.plot(audio) 
    # label the axes 
    plt.ylabel( "Amplitude" ) 
    plt.xlabel( "Time" ) 
    # set the title 
    # plt.title("Sample Wav") 
    # display the plot 
    plt.savefig( "C:/Users/work_computer/Desktop/NTNU/machine_learning/shiyerplot/" + baichuan_wav.split( '.' )[0] + '.png' ) 
    # plt.show() 
    plt.close( 'all' ) 

#%%
#Construct the MFCC image for deep learning method
import os 
import librosa
import librosa.feature
import librosa.display
import matplotlib.pyplot as plt
import numpy as np 

path="C:/Users/work_computer/Desktop/NTNU/"

count =0
#提取MFCC特征
def get_features(song,directory,plotname):  
    global count
    y,_ = librosa.load(song)
    mfcc = librosa.feature.mfcc(y)

    plt.figure(figsize=(10,4))
    librosa.display.specshow(mfcc,x_axis='time',y_axis='mel')
    plt.tight_layout()
    plt.savefig( path+"/deep_learning/test_data/"+directory + plotname.split( '.' )[0] + '.png' ) 
    plt.close( 'all' ) 
    count = count+1

    return 0

baichuan_plots = [] 
for (_,_,filenames) in os.walk( path+"baichuan_test/" ): 
    baichuan_plots.extend(filenames) 
    break 

for baichuan_plot in baichuan_plots: 
    get_features( path+"/baichuan_test/" + baichuan_plot,"baichuan/",baichuan_plot)


shiyer_plots = [] 
for (_,_,filenames) in os.walk(path+"shiyer_test/" ): 
    shiyer_plots.extend(filenames) 
    break 

for shiyer_plot in shiyer_plots: 
    get_features( path+"/shiyer_test/" + shiyer_plot,"shiyer/",shiyer_plot)


zhou_plots = [] 
for (_,_,filenames) in os.walk(path+"zhou_test/" ): 
    zhou_plots.extend(filenames) 
    break 

for zhou_plot in zhou_plots: 
    get_features( path+"/zhou_test/" + zhou_plot,"zhou/",zhou_plot)


wu_plots = [] 
for (_,_,filenames) in os.walk(path+"wu_test/" ): 
    wu_plots.extend(filenames) 
    break 

for wu_plot in wu_plots: 
    get_features( path+"/wu_test/" + wu_plot,"wu/",wu_plot)


tang_plots = [] 
for (_,_,filenames) in os.walk(path+"tang_test/" ): 
    tang_plots.extend(filenames) 
    break 

for tang_plot in tang_plots: 
    get_features( path+"/tang_test/" + tang_plot,"tang/",tang_plot)

print(count)