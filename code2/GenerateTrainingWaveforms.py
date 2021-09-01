# -*- coding: utf-8 -*-
"""
Created on Wed May  5 11:22:06 2021

@author: kirti
"""
import time
import math
import numpy                     as np
import ahkab                     as ahkab 
from   matplotlib import pyplot  as plt
from   random     import randint


"""
if(validate_args()):
    square(cy, pe, pd, r, tm, fs ,dly)
else:
    print('args not correct')
"""
def validate_args(tm, cy, pe, dly):
    if dly + (cy * pe) > tm:
        return False
    else:
        return True


def expo(t,dl,pd):
     
     t_ = np.arange(0, ex_pd, (1/fs))
     
     ex = []
     ex = np.array(ex)
     
     cy = math.floor(pd/ex_pd)
     for i in range(cy):
         fn  = ahkab.time_functions.exp(0.2, 0.6, 0, 0.0003, ex_pd/4, 0.0007)
         fn  = np.frompyfunc(fn, 1, 1)
         ex_ = fn(t_)
         ex  = np.append(ex,ex_)
     
     ex = ex.flatten()
     a = int((dl)*fs)
     b = (len(t) - (int((dl)*fs) + (len(t_)*cy)))
     ex = np.pad(ex,(a,b), 'edge')
     return ex
 
"""
Sine
"""
def sine(t, f):
    s = 0.5 + 0.5 * np.array( np.sin( (2 * np.pi *  t) * f)  )
    return s              

"""
sine
"""
def sine_(t, f, pd, dl):   
    #create the time which is only multiple of 1/f
    ts = 1/f
    cy = math.floor(pd/ts)
    t_ = np.arange(0,(cy * ts),(1/fs))
     
    fn = ahkab.time_functions.sin(0.5, 0.5, f, td=0.0, theta=0.0, phi=270.0)
    fn = np.frompyfunc(fn, 1, 1)
    s = fn(t_)
    
    a = int(dl*fs)
    b = int(len(s))
    c = int(len(t) - (a+b))
    
    s = np.pad(s , ( a , c ))
    
    return s
    
"""
Square
"""                 
def square(cy, pe, pd, r, t, fs, dl):
    
    t_ = np.arange(0, cy*pe, (1/fs))
    
    fn = ahkab.time_functions.pulse(0, 1, 0, r, pd, r, pe)
    fn = np.frompyfunc(fn, 1, 1)
    sq_ = fn(t_)
       
    a = int(dl*fs)
    b = int(len(sq_))
    c = int(len(t) - (a+b))
    
    sq = np.pad(sq_ , ( a , c ))
    
    return sq            
    
def generate_examples(n,tm):
    
    t  = []
    X =  []
    y =  []
    sq = []
    
    t = np.arange(0, tm, (1/fs))
    
    for i in range(n):
        
        cy = randint(cy_ran_min,cy_ran_max) 
        pe = randint(pe_ran_min,pe_ran_max) / 1000
        pd = randint(du_ran_min,du_ran_max) / 100
        dl = randint(dl_ran_min,dl_ran_max) / 1000
        
        if False == validate_args(tm, cy, pe, dl+dl_y):
            print('error %f %f %f %f' %(tm, cy, pe, dl+dl_y))
        
        sq_ = square(cy, pe, pd*pe , r, t, fs, dl)
        sq  = np.append(sq,sq_)
        
        X_ = sine_(t,fc_X,pd*pe,dl)
        #X_  = sine(t, fc_X)
        X   = np.append(X,X_) 
        
        y_  = expo(t,dl+dl_y,pd*pe)
        y   = np.append(y,y_)
        
    #X = X * sq
    X = X.reshape(n,-1,1)
    y = y.reshape(n,-1)
    
    X= X.astype(np.int32)
    y= y.astype(np.int32)
    return t,X,y



fs         = 2000
r          = 0.001
cy_ran_min = 1
cy_ran_max = 1
du_ran_min = 50  
du_ran_max = 90  
pe_ran_min = 16 #ms 
pe_ran_max = 30 #ms
dl_ran_min = 5  #ms 
dl_ran_max = 10 #ms

dl_y       = 0.010
fc_X       = 200 

ex_pd      = 0.008

tm         = 0.05
n          = 6

a = time.time()
t,X,y      = generate_examples(n,tm)
b = time.time()

print(b-a)

X          = X.reshape(n,-1,1)
y          = y.reshape(n,-1,1)

print(y.shape)

for i in range(6):
    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.plot(t,y[i])
    ax1.set_ylim(-0.05,0.55)
    ax1.set_ylim(-0.1,1.1)
    ax1.legend(["Training output - AN1 approximation"])
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('amplitude (V)')
    
    ax2.plot(t,X[i])
    ax2.set_ylim(-0.05,0.55)
    ax2.set_ylim(-0.1,1.1)
    ax2.legend(["Training input - Modulated pulse "])
    ax2.set_xlabel('time (s)')
    ax2.set_ylabel('amplitude (V)')    

    #fig.set_size_inches((18, 11), forward=False)
    #plt.savefig('%d' %(i),figsize =() ,dpi=500)