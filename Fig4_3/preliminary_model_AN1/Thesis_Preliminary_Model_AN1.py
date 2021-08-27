# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 08:33:59 2021

@author: kirteekumar
"""

import wave
import pyaudio
import sys
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfft
from scipy.io import wavfile 
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,SimpleRNN
import tensorflow as tf
def get_samples(wf):
    freq        = 4800
    amplitude   = 1 
    accu        = 0
    fsample     = wf.getframerate()
    N           = wf.getnframes()
    step        = 2*np.pi*freq/fsample
    t           = np.arange(N)*step + accu
    accu        += N*step
    sine_wave   = amplitude * np.sin(t)
    return sine_wave

def trim_audio(start_time, end_time, df, wf):
    a = start_time * wf.getframerate()
    b = end_time   * wf.getframerate()
    return df.iloc[round(a):round(b)]

path               = 'C:/Users/kirti/Desktop/Thesis/FromProfessor/data/python/AN1/AN1/'
fileName           = 'AN1 1.wav'

wf                 = wave.open(path+fileName,'r')

#take all samples
samples = wf.readframes(wf.getnframes())

#samples are in byte form, convert them to numpy array of 16 bit integers
samples = np.frombuffer(samples, dtype=np.int16)

#seperate them into 2 columns
samples = np.reshape(samples,[-1,2])

#------------------------------------------------------------------------------
#Modulelate the pulses by sinewave
sine_wave           = get_samples(wf)
modulated_sine_wave = sine_wave * samples[:,1]  

#------------------------------------------------------------------------------
#create a data frame and give them lable
df = pd.DataFrame(samples ,columns =['depolr','pulses'])
df['pulses'] = modulated_sine_wave


#------------------------------------------------------------------------------
#take some portion for training
start_time = 0.4
end_time   = 0.5

df_train = trim_audio(start_time,end_time,df,wf)

#------------------------------------------------------------------------------
#scale the data
scaler = MinMaxScaler()
scaler.fit(df)      #Data is fit for whole wave

scaled_data = scaler.transform(df_train)
scaled_targt = scaled_data[:,[0]]
scaled_train = scaled_data[:,[1]]

#------------------------------------------------------------------------------
#Generate timeseries.
length      = 892
batch_size  = 100
generator   = TimeseriesGenerator(scaled_train, scaled_targt, length=length, batch_size=batch_size)

#------------------------------------------------------------------------------
#Generate model.

n_features = 1

# define model
model = Sequential()

# Simple RNN layer
model.add(LSTM(length,input_shape=(length, n_features)))

# Final Prediction
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

# fit model
model.fit_generator(generator,epochs=5)

model.history.history.keys()
losses = pd.DataFrame(model.history.history)
losses.plot()

#------------------------------------------------------------------------------
#Evaluate model.
#take some portion for testing
start_time = 0.4
end_time   = 0.5


df_test             = trim_audio(start_time,end_time,df,wf)

#We are predicting values ahead. so substract the length.
c = len (df_test)
d = c - length      

scaled_test         = scaler.transform(df_test)
scaled_test_target   = scaled_test[:,[0]]
scaled_test         = scaled_test[:,[1]]

test_predictions = []

print('Predicting results...')


for i in range(d):
    print (i)
    a = i
    b = i + length 
    current_batch = scaled_test[a:b]
    current_batch = current_batch.reshape((1, length, n_features))
    
    current_pred = model.predict(current_batch)[0]
    
    # store prediction
    test_predictions.append(current_pred) 
    
    
#------------------------------------------------------------------------------   
# Create plots with pre-defined labels.
plt.plot(scaled_test)
plt.plot(scaled_test_target)
plt.plot(test_predictions)  
plt.legend(["Test sequence","expected result ", "predicted result"]) 



if 0:
    plt.plot(test_predictions)
    plt.plot(scaled_test_target)
    plt.plot(scaled_test)
    plt.legend(["test predictions","expected result ", "test input"])
    plt.ylabel('amplitude')
    plt.xlabel('time steps')

#tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)






