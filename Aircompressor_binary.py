import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense,LSTM,Conv1D,Input
import pandas as pd
import sys
import matplotlib.pyplot as plt
from scipy.stats import skew,kurtosis
from scipy.signal import find_peaks

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

file='AirCompressor_Data'


def clipping(directory,name):
    data_arr=[[] for x in range(len(directory))]
    count=0
    for l in directory:
        files=open(file+'\\'+name+'\\'+l)
        data=[float(x) for x in files.read().split(',')]

        length=len(data)
        split=[[] for x in range(9)]
        stds=[]
        for n in range(9):
            split[n]=np.array(data[n*int(length/10):(n+2)*int(length/10)])
            stds.append(np.std(split[n]))
        min_std=np.min(stds)
        indx=stds.index(min_std)
        data_arr[count]=split[indx]
        count+=1
    return np.array(data_arr)

dir= os.listdir(file+'\Healthy')
healthy=clipping(dir,'Healthy')

dir2=os.listdir(file+'\LIV')
LIV=clipping(dir2,'LIV')

dir3=os.listdir(file+'\LOV')
LOV=clipping(dir3,'LOV')


dir4=os.listdir(file+'\Bearing')
Bearing=clipping(dir4,'Bearing')


dir5=os.listdir(file+'\Flywheel')
Flywheel=clipping(dir5,'Flywheel')




print(LIV.shape,healthy.shape)

LIV=pd.DataFrame(data=LIV)
LIV['label']=0


healthy=pd.DataFrame(data=healthy)
healthy['label']=1



frames=[LIV,healthy]
df=pd.concat(frames)
print(df.shape)

df.index=list(range(len(df)))
df=df.reindex(np.random.permutation(df.index))
print(df.head())
labels=df.pop('label')
print(np.sum(labels))
df=(df-df.mean())/df.std()
print(df)

def feat_extract(data):
    new_data=[]
    for n in data:
        stuff=[]
        stuff.append(np.mean(n))
        stuff.append(np.max(n))
        stuff.append(np.var(n))
        stuff.append(np.sqrt(np.mean(np.square(n))))
        stuff.append((np.sqrt(np.mean(np.square(n))))/np.mean(n))
        stuff.append(np.max(np.abs(n))/np.sqrt(np.mean(np.square(n))))
        stuff.append(kurtosis(n))
        stuff.append(skew(n))
        peak,_=find_peaks(n)
        for x in sorted(n[peak])[-30:]:
            stuff.append(list(n[peak]).index(x))
            
        
        new_data.append(stuff)
    return np.array(new_data)

Feat=feat_extract(df.values)

print('feat: ',Feat.shape)

def bin(data):
    bin_data=[]
    for n in data:
        bins=[]
        indx=int(len(n)/8)
        for x in range(8):
            bins.append(np.sum(n[x*indx:(x+1)*indx]))
        Sum=np.sum(bins)
        bin_data.append(bins/Sum)
    return np.array(bin_data)

    
    

from scipy import fftpack
y_fft = fftpack.fft(df)
y_fft=np.abs(y_fft)
data=feat_extract(y_fft)
bin_data=bin(y_fft)
DATA=np.concatenate((Feat,data,bin_data),axis=1)
tsplit=0.2
train,test=np.split(DATA,[int(len(DATA)*(1-tsplit))])
train_labels,test_labels=np.split(labels,[int(len(DATA)*(1-tsplit))])

print(train.shape,test.shape)
#PCA
from sklearn.decomposition import PCA
#y_fft=np.array(y_fft).reshape(-1,10000)
#print(y_fft)
#n_com=286
#pca=PCA(n_components=n_com)
#train_pca=pca.fit_transform((y_fft))
#print(train_pca.shape)

#train_pca=data.reshape(-1,256,1)
#print(train_pca.shape)

model=tf.keras.models.Sequential([
    #tf.keras.layers.Conv1D(32,100,activation='relu',input_shape=[None,1]),
    #tf.keras.layers.LSTM(64, return_sequences=True,input_shape=[None,1]),
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
    ])
model.compile(optimizer='adam',metrics=['accuracy'],loss='binary_crossentropy')
history=model.fit(train,train_labels,validation_data=(test,test_labels),epochs=50,batch_size=10,verbose=2)
epoch=history.epoch
data=pd.DataFrame(history.history)


