import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import tensorflow as tf
import os

#find contours and crop 
def process_img(dir,label):
    images=os.listdir(dir)
    output=[[]]*len(images)
    for i in range(len(images)):
        image=cv2.imread(dir+images[i],cv2.IMREAD_GRAYSCALE)
        ret,thresh=cv2.threshold(image,127,255,cv2.THRESH_BINARY_INV)
        contors,h=cv2.findContours(image=thresh,mode=cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_NONE)
        for cnt in contors:
            x,y,w,h=cv2.boundingRect(cnt)
        crop=image[y:y+h,x:x+w]
        resized=cv2.resize(crop,(28,28))
        output[i]=resized
    return np.array(output)


one=process_img('extracted_images/1/',1)
two=process_img('extracted_images/2/',2)
three=process_img('extracted_images/3/',3)
four=process_img('extracted_images/4/',4)
five=process_img('extracted_images/5/',5)
six=process_img('extracted_images/6/',6)
seven=process_img('extracted_images/7/',7)
eight=process_img('extracted_images/8/',8)
nine=process_img('extracted_images/9/',9)
plus=process_img('extracted_images/+/',10)
minus=process_img('extracted_images/-/',11)

min_len=min(len(one),len(two),len(three),len(four),len(five),len(six),len(seven),len(eight),len(nine),len(plus),len(minus))

one=one[:min_len,:,:]
two=two[:min_len,:,:]
three=three[:min_len,:,:]
four=four[:min_len,:,:]
five=five[:min_len,:,:]
six=six[:min_len,:,:]
seven=seven[:min_len,:,:]
eight=eight[:min_len,:,:]
nine=nine[:min_len,:,:]
minus=minus[:min_len,:,:]
plus=plus[:min_len,:,:]

#create labels

print(plus.shape,min_len)
ones=np.ones([min_len,1])
labels=ones
for n in range(1,11):
    labels=np.concatenate((labels,ones+n),axis=0)


#combine and shuffle data
data=np.concatenate((one,two,three,four,five,six,seven,eight,nine,minus,plus),axis=0)

p = np.random.permutation(len(data))
shuffled_data=data[p]
shuffled_labels=labels[p]
shuffled_data=np.reshape(shuffled_data,[len(shuffled_data),28,28,1])
print(shuffled_labels)
print(shuffled_labels.shape,shuffled_data.shape)

#test and train split
train_split=int(0.8*len(shuffled_data))
train=shuffled_data[:train_split,:,:]
train_labels=shuffled_labels[:train_split]
test=shuffled_data[train_split:,:,:]
test_labels=shuffled_labels[train_split:]
print(train.shape,train_labels.shape,test.shape,test_labels.shape)
#cnn

model=tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(30,5,input_shape=(28,28,1),activation='relu'),
    tf.keras.layers.MaxPool2D(2),
    tf.keras.layers.Conv2D(12,3,activation='relu'),
    tf.keras.layers.MaxPool2D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(126,activation='relu'),
    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.Dense(13,activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

mc = tf.keras.callbacks.ModelCheckpoint('handwritten_symbols_classification_model.h5', monitor='val_loss', mode='min', verbose=1,save_best_only=True)
model.fit(train,train_labels,epochs=15,batch_size=500,verbose=2,validation_data=(test,test_labels),callbacks=[mc])