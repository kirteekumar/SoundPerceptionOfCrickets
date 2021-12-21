# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 00:53:40 2021

@author: Kirteekumar Sonawane
"""

import os
import pickle
import time
import wave
import psutil
import pyaudio
import datetime
import numpy                    as np
import pandas                   as pd 
import seaborn                  as sns
import tensorflow               as tf
import sklearn                  as sk
from tensorflow import keras
from matplotlib import pyplot as plt
from time_series_generator import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from random import randint
from tensorflow.keras.layers import Bidirectional,TimeDistributed


#==============================================================================
current_path = os.getcwd()

#==============================================================================
TRAIN               = 1
LOAD_MODELS         = 1
PREDICT             = 1
REAL_TIME_PREDICT   = 1

#Parameters====================================================================
fs                  = 20000     #Sampling frequency for orininal wave
r                   = 10        #Sampling rate reduction factor 
fs_downsampled      = fs/r
modulating_freq     = 200       #Modulating frequency for input pulses
t1                  = 0.0       #Start time for training
t2                  = 18.0       #End time for training
lstm_units          = 100        
timesteps           = 200       
epochs              = 500     
verbose             = 1
batch_size          = 1

#Functions=====================================================================
def expand_data_by_timeseries_generator(X,timesteps):
        #Timeseries - it will multiply total data by "timestaps" times 
        #e.g. 200 times if timesteps are 200
        # define generator
        source = list()
        generator_source = tf.keras.preprocessing.sequence.TimeseriesGenerator(X.flatten(), 
                                        X.flatten(),length=timesteps,  batch_size=1)
        
        for i in range(len(generator_source)):
           tmp1, tmp = generator_source[i]
           source.append(tmp1)
           
        source = np.array(source)
        source = source.flatten()
        source = source.reshape(-1,1)
        return source

#==============================================================================
def truncate_data_and_reshape(X,y,timesteps,n_features):
    new_size = int(X.size/n_features - (X.size/n_features)%timesteps)
    tmp = np.zeros([n_features,new_size])
    
    for i in range(n_features):
        #we need to remove the 0 because of dimension.
        if n_features==1:
            tmp[i,:] = X[0:int(X.size/n_features - (X.size/n_features)%timesteps),i]
        elif n_features==2:
            tmp[i,:] = X[i,0:int(X.size/n_features - (X.size/n_features)%timesteps),0] #better way?   
        y = y[0:(len(y) - len(y)%timesteps)] #Do you know any better way??
    
    X = tmp
    X = X.reshape(-1,timesteps,n_features)  #X in 3D [batches,timesteps,n_feature]
    y = y.reshape(-1,timesteps)             #y [batches,timesteps]
    return X,y

#==============================================================================
def scale_data(a): 
    scaler = MinMaxScaler()
    a = a.reshape(-1,1)
    a = scaler.fit_transform(a)
    return a,scaler

#==============================================================================
def modulate(X,m):
    ts              = 1/fs_downsampled
    n               = len(X)
    t               = np.arange(0,ts * n, ts)
    s               = 0.5 + 0.5 * np.sin(2 * np.pi * m * t )
    X_modulated    = s * X
    return t,X_modulated

#==============================================================================
def trim_and_save(p,q,t1,t2):
    with wave.open(p,'rb') as a:
        p           = a.getparams()
        fr          = fs
        nf          = p.nframes
        f           = a.readframes(nf)
        f           = f[  (int(t1*fr)*4) : (int(t2*fr)*4) ]
    
    with wave.open(q,'wb') as b:
        b.setparams(p)
        b.setframerate(fs)
        b.writeframes(f)
        
#==============================================================================
def get_wave_data(q,r):
    with wave.open(q,'rb') as a:
        p           = a.getparams()
        nf          = p.nframes
        f           = a.readframes(nf)
        f           = np.frombuffer(f, dtype=np.int16)
        f           = np.reshape(f,[-1,2])   
        y,X         = f[0::r,0] , f[0::r,1]
        
    return X,y

#==============================================================================
def truncate_reshape_for_plotting(a,X):
    a = a[0 : X.size - int(X.size % timesteps)  ]
    a = a.reshape(-1)
    return a


#code==========================================================================
p    = current_path+r'\original_waveforms\AN1 1.wav'
q1   = current_path+r'\trimmed_waveforms\AN1 1_.wav'
trim_and_save(p,q1,t1,t2)

p    = current_path+r'\original_waveforms\LN2 2.wav'
q2   = current_path+r'\trimmed_waveforms\LN2 2_.wav'
trim_and_save(p,q2,t1,t2)

p    = current_path+r'\original_waveforms\LN3 3.wav'
q3   = current_path+r'\trimmed_waveforms\LN3 3_.wav'
trim_and_save(p,q3,t1,t2)

p    = current_path+r'\original_waveforms\LN4 1.wav'
q4   = current_path+r'\trimmed_waveforms\LN4 1_.wav'
trim_and_save(p,q4,t1,t2)

p    = current_path+r'\original_waveforms\LN5 1.wav'
q5   = current_path+r'\trimmed_waveforms\LN5 1_.wav'
trim_and_save(p,q5,t1,t2)


X   ,y1 = get_wave_data(q1,r)
tmp ,y2 = get_wave_data(q2,r)
tmp ,y3 = get_wave_data(q3,r)
tmp ,y4 = get_wave_data(q4,r)
tmp ,y5 = get_wave_data(q5,r)

   
#==========================================================================
#Modulate the input pulses
t,X = modulate(X,modulating_freq)

#==========================================================================
#Convert input and output data in 0 to 1 range.
X,scaler_X          = scale_data(X)
y1,scaler_y1        = scale_data(y1)
y2,scaler_y2        = scale_data(y2)
y5,scaler_y5        = scale_data(y5)
y3,scaler_y3        = scale_data(y3)
y4,scaler_y4        = scale_data(y4)

#==========================================================================
#Plot the training data
if 0:
    fig, ((ax11), (ax21), (ax31), (ax41), (ax51), 
          (ax61)) = plt.subplots(6,1, sharex=True)
    
    fig.suptitle('Training waveforms')
    ax11.plot(t, y4)
    ax11.set_title('LN4')
    ax21.plot(t, y3)
    ax21.set_title('LN3')
    ax31.plot(t, y5)
    ax31.set_title('LN5')
    ax41.plot(t, y2)
    ax41.set_title('LN2')
    ax51.plot(t, y1)
    ax51.set_title('AN1')
    ax61.plot(t, X)
    ax61.set_title('Modulated pulses')
    plt.show()

#==========================================================================
dumy = X
tmp,tmp,X ,X_ ,tmp,tmp=train_test_split(t,X ,dumy,train_size=0.8,test_size=0.2,shuffle=False)
tmp,tmp,y1,y1_,tmp,tmp=train_test_split(t,y1,dumy,train_size=0.8,test_size=0.2,shuffle=False)
tmp,tmp,y2,y2_,tmp,tmp=train_test_split(t,y2,dumy,train_size=0.8,test_size=0.2,shuffle=False)
tmp,tmp,y3,y3_,tmp,tmp=train_test_split(t,y3,dumy,train_size=0.8,test_size=0.2,shuffle=False)
tmp,tmp,y4,y4_,tmp,tmp=train_test_split(t,y4,dumy,train_size=0.8,test_size=0.2,shuffle=False)
tmp,tmp,y5,y5_,tmp,tmp=train_test_split(t,y5,dumy,train_size=0.8,test_size=0.2,shuffle=False)


if TRAIN==1:    
    #since we will need X,y1,y2.. etc. later save them in another variable
    X_ex  = X
    y1_ex = y1
    y2_ex = y2
    y3_ex = y3
    y4_ex = y4
    y5_ex = y5
    
    #==========================================================================
    X_in     ,y1_out  = truncate_data_and_reshape(X_ex ,y1_ex,timesteps,1)
    y1_in    ,y2_out  = truncate_data_and_reshape(y1_ex,y2_ex,timesteps,1)
    y2_in    ,y5_out  = truncate_data_and_reshape(y2_ex,y5_ex,timesteps,1)
    y1_y5_in ,y3_out  = truncate_data_and_reshape(np.array([y1_ex,y5_ex]),y3_ex,timesteps,2)
    y5_y3_in ,y4_out  = truncate_data_and_reshape(np.array([y5_ex,y3_ex]),y4_ex,timesteps,2)
    
    X_in_     ,y1_out_  = truncate_data_and_reshape(X_,y1_,timesteps,1)
    y1_in_    ,y2_out_  = truncate_data_and_reshape(y1_,y2_,timesteps,1)
    y2_in_    ,y5_out_  = truncate_data_and_reshape(y2_,y5_,timesteps,1)
    y1_y5_in_ ,y3_out_  = truncate_data_and_reshape(np.array([y1_,y5_]),y3_,timesteps,2)
    y5_y3_in_ ,y4_out_  = truncate_data_and_reshape(np.array([y5_,y3_]),y4_,timesteps,2)
    
    #==========================================================================
    Data_For_Training   = [[X_in,y1_out],[y1_in,y2_out],[y2_in,y5_out],
                         [y1_y5_in,y3_out],
                         [y5_y3_in,y4_out]] #In form of [input, output]
    
    
    Data_For_Validation = [[X_in_,y1_out_],[y1_in_,y2_out_],[y2_in_,y5_out_],
                         [y1_y5_in_,y3_out_],
                         [y5_y3_in_,y4_out_]]
    
    
    neuron_names = ["AN1_1","LN2_2","LN5_1","LN3_3","LN4_1"]
    n_features          = np.array([1,5])
    n_features          = [1,1,1,2,2]
    
    total_start_time_t = time.time()
    for i in range(len(neuron_names)):   
        #Build the model =====================================
        model = tf.keras.Sequential()
        model.add(Bidirectional(tf.keras.layers.LSTM(lstm_units), 
        input_shape = (timesteps,n_features[i])))
        model.add((tf.keras.layers.Dense(timesteps, activation= 'relu' )))
        model.compile(loss= 'huber_loss' , optimizer= 'adam' , metrics=[ 'accuracy'])
        print(model.summary())
           
        #Prepare Data for Training=============================================
        print('=========================================================')
        print('Training the model....')
        print('=========================================================')
        start_time       = time.time()
        history = model.fit(Data_For_Training[i][0],Data_For_Training[i][1],
                  validation_data=(Data_For_Validation[i][0],Data_For_Validation[i][1]),
                              epochs=epochs,verbose=verbose,batch_size=batch_size)
        end_time         = time.time()
        training_time    = end_time - start_time
        
        print('=========================================================')
        print('Testing the model accuracy....')
        print('=========================================================')
        evaluation = model.evaluate(Data_For_Validation[i][0],Data_For_Validation[i][1])

        print('=========================================================')
        print('Saving the model and parameters....')
        print('=========================================================')    
        model.save(current_path+r'\saved_models_and_params\{}'.format(neuron_names[i]))
        with open(current_path+r'\saved_models_and_params\{}\saved_parameters_{}'
                  .format(neuron_names[i] , neuron_names[i]), "wb") as  fp:
             dictionary = {
              "training_time"            : training_time,
              "evaluation"               : evaluation,
              "history_history"          : history.history,
              "t1"                       : t1,
              "t2"                       : t2,
              "fs"                       : fs,
              "r"                        : r,
              "modulating_freq"          : modulating_freq,
              "lstm_units"               : lstm_units,
              "timesteps"                : timesteps,
              "epochs"                   : epochs,
              "verbose"                  : verbose,
              "batch_size"               : batch_size}
              
             pickle.dump(dictionary       , fp)
    
    total_end_time_t = time.time()
    total_training_time = total_end_time_t-total_start_time_t
    print("Total time to train all neurons")
    print(total_training_time)
    
if LOAD_MODELS==1:
    total_start_time_l = time.time()
    str_p = r'\saved_models_and_params'
    
    print('=========================================================')
    print('Loading the models....')
    print('=========================================================')  
    model1_an1 = keras.models.load_model(current_path+str_p+r'\AN1_1')
    model2_ln2 = keras.models.load_model(current_path+str_p+r'\LN2_2')
    model3_ln5 = keras.models.load_model(current_path+str_p+r'\LN5_1')
    model4_ln3 = keras.models.load_model(current_path+str_p+r'\LN3_3')
    model5_ln4 = keras.models.load_model(current_path+str_p+r'\LN4_1')   
    
    total_end_time_l = time.time()
    total_loading_time = total_end_time_l-total_start_time_l
    print("Total time to load all models")
    print(total_loading_time)


if PREDICT==1:
    print('=========================================================')
    print('Making the predictions....')
    print('=========================================================') 
    total_start_time_p = time.time()
    X_       , tmp  = truncate_data_and_reshape(X,y1,timesteps,1)
    yhat1           = model1_an1.predict(X_)
    
    yhat1_          = yhat1.reshape(-1,1)
    yhat1_   , tmp  = truncate_data_and_reshape(yhat1_,yhat1_,timesteps,1)
    yhat2           = model2_ln2.predict(yhat1_.reshape(-1,timesteps,1))
    
    yhat2_          = yhat2.reshape(-1,1)
    yhat2_   , tmp  = truncate_data_and_reshape(yhat2_,yhat2_,timesteps,1)
    yhat5           = model3_ln5.predict(yhat2_.reshape(-1,timesteps,1))
    
    yhat1_          = yhat1.reshape(-1,1)
    yhat5_          = yhat5.reshape(-1,1)
    yhat3_   , tmp  = truncate_data_and_reshape(np.array([yhat1_,yhat5_]),X,timesteps,2)
    yhat3           = model4_ln3.predict(yhat3_.reshape(-1,timesteps,2))
    
    yhat3_          = yhat3.reshape(-1,1)
    yhat5_          = yhat5.reshape(-1,1)
    yhat4_   , tmp  = truncate_data_and_reshape(np.array([yhat3_,yhat5_]),X,timesteps,2)
    yhat4           = model5_ln4.predict(yhat4_.reshape(-1,timesteps,2))
    
    total_end_time_p = time.time()
    total_prediction_time = total_end_time_p-total_start_time_p
    print("Total time to predict whole data")
    print(total_prediction_time)
    
    X_p = truncate_reshape_for_plotting(X,X_)    
    t_p = truncate_reshape_for_plotting(t,X_)    
    y1_p = truncate_reshape_for_plotting(y1,X_)    
    y2_p = truncate_reshape_for_plotting(y2,X_)
    y5_p = truncate_reshape_for_plotting(y5,X_)    
    y3_p = truncate_reshape_for_plotting(y3,X_)   
    y4_p = truncate_reshape_for_plotting(y4,X_)          
    yhat1_p = truncate_reshape_for_plotting(yhat1,X_)    
    yhat2_p = truncate_reshape_for_plotting(yhat2,X_)
    yhat5_p = truncate_reshape_for_plotting(yhat5,X_)    
    yhat3_p = truncate_reshape_for_plotting(yhat3,X_)   
    yhat4_p = truncate_reshape_for_plotting(yhat4,X_)   
    
    fig, ((ax11, ax12), (ax21, ax22), (ax31, ax32), (ax41, ax42), (ax51, ax52), 
          (ax61, ax62)) = plt.subplots(6,2, sharex=True)
    
    fig.suptitle('Predictions of Neural network')
    ax11.plot(t_p, yhat4_p)
    ax12.plot(t_p, y4_p)
    ax21.plot(t_p, yhat3_p)
    ax22.plot(t_p, y3_p)
    ax31.plot(t_p, yhat5_p)
    ax32.plot(t_p, y5_p)
    ax41.plot(t_p, yhat2_p)
    ax42.plot(t_p, y2_p)
    ax51.plot(t_p, yhat1_p)
    ax52.plot(t_p, y1_p)
    ax61.plot(t_p, X_p)
    ax62.plot(t_p, X_p)
    
    ax11.set_title('LN4 - Predicted',pad=0.75)
    ax12.set_title('LN4 - Expected',pad=0.75)
    ax21.set_title('LN3 - Predicted',pad=0.75)
    ax22.set_title('LN3 - Expected',pad=0.75)
    ax31.set_title('LN5 - Predicted',pad=0.75)
    ax32.set_title('LN5 - Expected',pad=0.75)
    ax41.set_title('LN2 - Predicted',pad=0.75)
    ax42.set_title('LN2 - Expected',pad=0.75)
    ax51.set_title('AN1 - Predicted',pad=0.75)
    ax52.set_title('AN1 - Expected',pad=0.75)
    ax61.set_title('Modulated pulses',pad=0.75)
    ax62.set_title('Modulated pulses',pad=0.75)
    
    ax61.set_xlabel('time (s)')
    ax62.set_xlabel('time (s)')

    if 0:    
        fig, (ax_1, ax_2, ax_3) = plt.subplots(3,1,sharex=True)
        ax_3.plot(t_p,X_p)
        ax_2.plot(t_p,y1_p)
        ax_1.plot(t_p,yhat1_p)
        ax_3.set_xlabel('time (s)')
        ax_3.set_title('Modulated pulses',pad=0.75)
        ax_2.set_title('AN1 - Expected',pad=0.75)
        ax_1.set_title('AN1 - Predicted',pad=0.75)
        fig.suptitle('Predictions of bidirectional model for AN1 neuron')
       
if REAL_TIME_PREDICT==1:
    scaler_i = MinMaxScaler()
    scaler_i.fit(np.arange(-32768,32767).reshape(-1,1))
    
    #==========================================================================
    print('=========================================================')
    print('Predicting from the microphone input....')
    print('=========================================================')
    
    fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6)
    ax1.set_title('LN4 - Predicted',pad=0.75)
    ax2.set_title('LN3 - Predicted',pad=0.75)
    ax3.set_title('LN5 - Predicted',pad=0.75)
    ax4.set_title('LN2 - Predicted',pad=0.75)
    ax5.set_title('AN1 - Predicted',pad=0.75)
    ax6.set_title('Modulated pulses',pad=0.75)
    
    ax6.set_xlabel('time (samples)')
    
    CHUNK = 1000    
    WIDTH = 2
    CHANNELS = 1
    RATE = 2000
    RECORD_SECONDS = 5
    
    log = []
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(WIDTH),
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    output=True,
                    frames_per_buffer=CHUNK)

    
while True: 
        data = stream.read(CHUNK)
        
        frs = time.time()
        #======================================================================
        #convert the data for prediction
        #from bytestream to int16
        data = np.frombuffer(data, dtype=np.int16)
        
        data_ = data/32768
        
        #convert between 0 to 1
        #data_  = scaler_i.transform(data.reshape(-1, 1))
        #reshape for prediction
        data_  = np.reshape(data_,(5,200,1))
  
        #======================================================================
        #predict the output
        yhat1_          = model1_an1.predict(data_)
        yhat1           = np.array(yhat1_)
        
        yhat2           = model2_ln2.predict(yhat1.reshape(-1,timesteps,1))
        
        yhat5           = model3_ln5.predict(yhat2.reshape(-1,timesteps,1))
        
        yhat3           = np.array([yhat1,yhat5])
        yhat3           = model4_ln3.predict(yhat3.reshape(-1,timesteps,2))
        
        yhat4           = np.array([yhat3,yhat5])
        yhat4           = model5_ln4.predict(yhat4.reshape(-1,timesteps,2))
        
        #======================================================================
        ax6.set_ylim(0,1)
        ax5.set_ylim(0,1)
        ax4.set_ylim(0,1)
        ax3.set_ylim(0,1)
        ax2.set_ylim(0,1)
        ax1.set_ylim(0,1)
        
        # ax1.plot(yhat4[0,:] ,color='b')     
        # ax2.plot(yhat3[0,:] ,color='b')    
        # ax3.plot(yhat5[0,:] ,color='b')
        # ax4.plot(yhat2[0,:] ,color='b')
        # ax5.plot(yhat1[0,:] ,color='b')
        # ax6.plot(data_[0,:,0],color='b')
        
        
        ax1.plot(yhat4.flatten() ,color='b')     
        ax2.plot(yhat3.flatten() ,color='b')    
        ax3.plot(yhat5.flatten() ,color='b')
        ax4.plot(yhat2.flatten() ,color='b')
        ax5.plot(yhat1.flatten() ,color='b')
        ax6.plot(data_.flatten(),color='b')
        
        plt.pause(0.000000001)
        ax6.cla()
        ax5.cla()
        ax4.cla()
        ax3.cla()
        ax2.cla()
        ax1.cla()
        
        ax1.set_title('LN4 - Predicted',pad=0.75)
        ax2.set_title('LN3 - Predicted',pad=0.75)
        ax3.set_title('LN5 - Predicted',pad=0.75)
        ax4.set_title('LN2 - Predicted',pad=0.75)
        ax5.set_title('AN1 - Predicted',pad=0.75)
        ax6.set_title('Modulated pulses',pad=0.75)
        
        ax6.set_xlabel('time (s)')
    
        las = time.time()
        
        #This log contais the total time taken for prediction and plotting
        log.append(las-frs)
