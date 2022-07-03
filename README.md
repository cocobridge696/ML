# Machine Learning Projects
A few self-contained independent machine learning projects.
## Self driving car algorithms
File: self_driving_car.ipynb

Date completed: June 2022

A comparison of CNNs,Ensemble CNNS, SVMs,Decision trees and autoencoder alogirthms for predicting speed and steering angle from a still frame image taken from a PiCar. The Ensemble CNNs performed the best followed by the single CNN and autoencoder models. Hyperparameters were tuned using a bayesian optimisation scheme (tree partizan model).

## Black Jack reinforcement learning agent
File: BlackJack_Agent.ipynb

Date completed: December 2021

The game of blackjack may be mathematically formalized as an episodic finite Markov Decision Process (MDP). Reinforcement learning agents navigate the states of an MDP, by taking actions in each state under a given policy. This policy is altered to maximize a user-defined reward. Here we apply and compare Q Learning, Dynamic Programming and Monte Carlo methods to a stylised form of Blackjack without an opponent. These methods are applied to both an infinite and finite number of Decks. We show that in the case of infinite decks, the Dynamic Programming method outperforms both the Q Learning and Monte Carlo methods, which both follow a similar learning path and converge to similar values. We demonstrate that in the case of finite decks, an agent which retains a memory of cards played outperforms our old DP agent.


## Air Compressor Control monitoring
File: Aircompressor_binary_classification.py

Date completed: April 2020

This project involves using an acoustic dataset of compressors to determine if the air compressor is faulty. This is an example of using machine learning for control monitoring and it is becoming an important part of industrial processes at it can allow for early detection of machinery faults which can prevent heavy losses occurring due to process shutdown and machinery replacement. The model is a two dense layer binary classifier that used features extracted from the acoustic recordings. Before the features were extracted the acoustic recordings were clipped by splitting the recording into 9 parts and keeping the part that had the lowest standard deviation. This allowed for faster computation of the data and was essential to removing anomalous data. Then the data was normalized using z-score normalization and this was done to prevent exploding gradients during model training. Exploding gradients is common in training with unnormalized data as the data can create large losses based on the units of the data thus causing massive changes in trainable weights. Features were extracted in both the time and the frequency domain. The addition of Frequency domain analysis is essential in signal processing as it allows the signals energy over a range of frequencies to be observed. The features within frequency domain signals are not visible in the time domain. The time domain acoustic signal was converted to the frequency domain by Fourier transform. The following features were extracted from each acoustic signal in both the time and frequency domain: mean, max, min, variance, kurtosis, skew, top 30 peak locations, top 30 peak heights, crest factor, shape factor and root-mean-square. In addition to this the frequency data was split into 8 equal segment bins and the ratio of the individual bin energy to the total energy was used to give 8 more features. These feature extractions are typical for signal data due to them showing great performance in classifiers. The trained model performed well with these features achieving a 97.8% accuracy at distinguishing between a healthy and unhealthy compressor on the validation dataset

## Wind Speed prediction model 
File: Design_Project_Wind_Prediction.ipynb

Date completed: April 2021

This project was done to provide a prediction of windspeed that could be used to control processes relying on wind energy such as green ammonia production. The specific model architecture for this report follows a recurrent neural network (RNN) design using long-term short-term memory (LSTM) cells. RNNs are typically used with time series data in order to predict future values. LSTM cells contain more trainable parameter when compared to other typical RNN cells and therefore are more computationally expensive. Although, LSTM cells were still chosen due to the amount of data available being relatively large thus favouring LSTMs, due to them having more trainable parameters which are able to learn the trends in the data. In smaller datasets this can become undesirable as with more parameter’s models are prone to overfitting the data which essentially means the model is memorizing the training data and therefore when testing the model on validation data a large error is achieved. 
The model was trained on 40,600 wind data points from an accessible European wind dataset. The model was then validated against 9,930 data points from a separate section of the same dataset. The final model gave a normalised mean squared error of 0.0306. This error can be visualised by plotting the models predicted wind speeds for a range of the validation data points
