# Machine Learning Projects
A few self-contained independent machine learning projects including autoencoding, classification, regression time series analysis, natural language processing and clustering.

Most can be opened and ran in google colab.

## Air Compressor Control monitoring
File: Aircompressor_binary_classification.py

Date completed: April 2020

This project involves using an acoustic dataset of compressors to determine if the air compressor is faulty. This is an example of using machine learning for control monitoring and it is becoming an important part of industrial processes at it can allow for early detection of machinery faults which can prevent heavy losses occurring due to process shutdown and machinery replacement. The model is a two dense layer binary classifier that used features extracted from the acoustic recordings. Before the features were extracted the acoustic recordings were clipped by splitting the recording into 9 parts and keeping the part that had the lowest standard deviation. This allowed for faster computation of the data and was essential to removing anomalous data. Then the data was normalized using z-score normalization and this was done to prevent exploding gradients during model training. Exploding gradients is common in training with unnormalized data as the data can create large losses based on the units of the data thus causing massive changes in trainable weights. Features were extracted in both the time and the frequency domain. The addition of Frequency domain analysis is essential in signal processing as it allows the signals energy over a range of frequencies to be observed. The features within frequency domain signals are not visible in the time domain. The time domain acoustic signal was converted to the frequency domain by Fourier transform. The following features were extracted from each acoustic signal in both the time and frequency domain: mean, max, min, variance, kurtosis, skew, top 30 peak locations, top 30 peak heights, crest factor, shape factor and root-mean-square. In addition to this the frequency data was split into 8 equal segment bins and the ratio of the individual bin energy to the total energy was used to give 8 more features. These feature extractions are typical for signal data due to them showing great performance in classifiers. The trained model performed well with these features achieving a 97.8% accuracy at distinguishing between a healthy and unhealthy compressor on the validation dataset

## Wind Speed prediction model 
File: Design_Project_Wind_Prediction.ipynb

Date completed: April 2021

This project was done to provide a prediction of windspeed that could be used to control processes relying on wind energy such as green ammonia production. The specific model architecture for this report follows a recurrent neural network (RNN) design using long-term short-term memory (LSTM) cells. RNNs are typically used with time series data in order to predict future values. LSTM cells contain more trainable parameter when compared to other typical RNN cells and therefore are more computationally expensive. Although, LSTM cells were still chosen due to the amount of data available being relatively large thus favouring LSTMs, due to them having more trainable parameters which are able to learn the trends in the data. In smaller datasets this can become undesirable as with more parameter’s models are prone to overfitting the data which essentially means the model is memorizing the training data and therefore when testing the model on validation data a large error is achieved. 
The model was trained on 40,600 wind data points from an accessible European wind dataset. The model was then validated against 9,930 data points from a separate section of the same dataset. The final model gave a normalised mean squared error of 0.0306. This error can be visualised by plotting the models predicted wind speeds for a range of the validation data points

## Sentiment Analysis
File: IMDB_sentiment_BoW,Conv1D,MultiCNN_Bclass.ipynb

Date completed: April 2020

For this project the sentiment of IMDB reviews were analysed using a range of methods including bag of words, 1D convolutions and a multi-channel convolution neural network. The bag of words section takes strong influence from TensorFlow tutorials and the natural language processing course on Coursera while the other methods are from a range of sources. The bag of words methods does not consider the order of the words (hence the name) while the other methods do consider the sequence of words. Yet, it turned out that the bag of words model achieved the highest accuracy. The following accuracies were achieved: BoW model acc=90%, CNN model acc=89%, MultiCNN acc=88%.

## Clustering using auto encoded features
File: MNIST_Aclust.ipynb

Date completed: April 2020

This project involved training an autoencoder to encode and decode data so that the input and output are as similar as possible. After this had been trained the encode and decode part of the model were separated and encoded data was passed into a custom clustering layer that used a t distribution to measure the similarity between the data and the centroids. The accuracy of the clustering was measures using the adjusted rand score, Munkres’ Assignment algorithm and Silhouette score.

## Bitcoin Price predictions
File: UV_Bitcoin_price_prediction.ipynb, MV_Bitcoin_price_prediction.ipynb

Date completed: August 2021

These projects were done to try and predict the future price of bitcoin using LSTMs and autoregressive integrated moving average (ARIMA) models. Models were constructed using both multi-variate and uni-variate data. Both models failed to predict bitcoins price better than simply using the previous datapoint. These projects were still uploaded as I found it interesting that for accurate predictions (especially when related to stocks) lots of different data could be relevant such as economic cycles and even social media and this data is often not easily accessible. Also, in depth knowledge about the data and what is relevant is required.

## Anomaly Detection
File: Weather_anomaly_detection_AE.ipynb

Date completed: August 2020

For this project a trained autoencoder was used to predict anomalous data. This is significant for detecting faults in machines, unusual process conditions or potential hazards. For this model the autoencoder was trained using 'normal' data and then when the model was tested, if the recreation error was larger than the max error for the training data then the point was classed as anomalous. The model seemed to perform well as shown by the final plot. The model can be tweaked to detect single anomalies or a series of anomalies. 

## Handwritten equation solver
File: Equation_Solver.ipynb, hand_wriiten_equation_model.py,handwritten_symbols_classification.h5

Date completed: September 2021

This project involved decomposing images into bounding boxes of symbols using cv2 and then classifying the image using a CNN. The classified images were then used to evaluate the answer to the handwritten equation.  Equation_Solver.ipynb is used to give the answer to the handwritten equation given the model: handwritten_symbols_classification.h5. The model was trained using hand_wriiten_equation_model.py.
