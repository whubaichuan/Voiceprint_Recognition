# Voiceprint Recognition
This is a demo of the voiceprint recognition based on my custom dataset. Here we use the machine and deep learning to make a comparison.

### The overview
We collect the voice from 5 people, who are "baichuan", "shiyer", "zhou", "wu", "tang". In the following step, you only need to train the model with 3 voices. We split the voice into 50 fragments and each fragment is 12s. The raw .wav files are in ./raw_data and you can use the [Preprocessing.py](https://github.com/whubaichuan/Voiceprint_Recognition/blob/main/raw_data/Preprocessing.py) to prepare your own custom dataset.

In ./machine_learning and ./deep_learnig, you could train and test the model separately. The workflow is illustrated in the following picture.

![avatar](https://github.com/whubaichuan/Voiceprint_Recognition/blob/main/image/flow.png)

### Feature extraction
There are many features can be extracted and fed into the method of machine learning such as Log-Mel, MFCC, D-vector (By Google), the spectrum diagram and time-frequency diagram. We choose the MFCC feature based on the auditory characteristics of humans.
![avatar](https://github.com/whubaichuan/Voiceprint_Recognition/blob/main/image/features.png)

### Machine Learning
There are many methods of machine learning such as GMM, HMM, SVM. We here use SVM due to that SVM perform well in solving small sample nonlinear classification.  
### Deep Learning
The spectrum diagram of MFCC is the data fed into a vanilla simple VGGNet here. 

>In the future, you can combine other features like the initial waveform figure, the harmonic spectrum, left & right channels and so on to improve the performance of this model. Besides, you could try RNN and LSTM to capture the time series information and aggregate the CNN and RNN together. 

### Experiments

By abundant experiments, we find that the accuracy, f1-score and the confusion matrix will change depending on the different small datasets. Thus, we conduct the following experiments.

> Near girl means the girl who spent most of her time with me. Far girl and far boy mean they are almost a stranger to me.

+ (a) When use "baichuan"(boy), "shiyer"(near girl) and "tang"(far girl)

|                  | machine_learning(train)       | machine_learning(test) | deep_learning(train)          | deep_learning(test) |
|------------------|-------------------------------|------------------------|-------------------------------|---------------------|
| accuracy         |              1.00             |          0.66          |              0.97             |         0.86        |
| f1-score         |              1.00             |            -           |              0.97             |          -          |
| confusion matrix | [15,0,0<br>0,15,0,<br>0,0,15] |            -           | [18,0,1<br>0,13,0,<br>0,0,13] |          -          |

+ (b) When use "baichuan"(boy), "wu"(far boy) and "tang"(far girl)

|                  | machine_learning(train)       | machine_learning(test) | deep_learning(train)          | deep_learning(test) |
|------------------|-------------------------------|------------------------|-------------------------------|---------------------|
| accuracy         |              0.86             |          0.73          |              0.86             |         0.53        |
| f1-score         |              0.86             |            -           |              0.86             |          -          |
| confusion matrix | [11,4,0<br>1,14,0,<br>0,1,14] |            -           | [14,5,0<br>0,13,0,<br>0,1,12] |          -          |

+ (c) When use "baichuan"(boy), "zhou"(far girl), "tang"(far girl)

|                  | machine_learning(train)       | machine_learning(test) | deep_learning(train)          | deep_learning(test) |
|------------------|-------------------------------|------------------------|-------------------------------|---------------------|
| accuracy         |              1.00             |          1.00          |              1.00             |         0.93        |
| f1-score         |              1.00             |            -           |              1.00             |          -          |
| confusion matrix | [15,0,0<br>0,15,0,<br>0,,15] |            -           | [19,0,0<br>0,13,0,<br>0,0,13] |          -          |

A visulization of training process of deep learning:
![avatar](https://github.com/whubaichuan/Voiceprint_Recognition/blob/main/image/cnn_plot.png)

### Results
1. We could see that the accuracy of (c) is higher than (a) and (b) not only in machine learning but also in deep learning. Does that mean that the near people and the gender will affect the model?
2. For (a) and (c), it's interesting that **if two people live together for a long time, whether the voiceprint of them is difficult to be recognized or not.**
3. For (b) and (c), we should consider **how to extract the effective feature related to gender to improve the performance.**
4. The method of deep learning is better than the method of machine learning in (a) and (c). But there is a difference in (b).
5. It's hard to say which method is better regarding to the above comparison with only 50 training data for each label. As we all know, deep learning will show ability with enormous data. Thus, I think more experiments with adequate datasets will tell us the truth.


It has been pointed out that:
> The application of voiceprint recognition has some disadvantages. For example, the voice of the same person is volatile and easily affected by physical condition, age, emotion, etc.; for example, different microphones and channels have an impact on recognition performance; for example, environmental noise interferes with recognition; Another example is that in the case of mixed speakers, it is not easy to extract the features of human voiceprints; ... etc.


### Acknowledge
For more details about the technologies and trendency of voiceprint recognition, please refer to [here](https://zhuanlan.zhihu.com/p/67563275.
)