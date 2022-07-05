# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 13:03:30 2022

@author: gcoffey
"""

''' Easy molecule function: 
    
easy molecule is modified from the easyR method of Sweeney & Burnham,
(1990) to use first order reaction kinetic parameters of molecular
reactions determined in the lab.  The modification collapses the multiple
reaction model to a single activation energy and frequency factor.  The
math for calculating reaction extents for non-isothermal heating steps is
taken directly from the easyR model.

References:

Burnham A. K. and Sweeney J. J. (1989) A chemical kinetic model of
  vitrinite maturation and reflectance. Geochimica et Cosmochimica
  Acta 53(10), 2649.

Sweeney J. J. and Burnham A. K. (1990) Evaluation of a simple model of
  vitrinite reflectance based on chemical kinetics. The American
  Association of Petroleum Geologists Bulletin 74(10), 1559-1570.

Usage
[F, Findiv, A, E, f, Iindividual] = easyA(time, temp, E, A, Iindividual)

INPUT (REQUIRED):
time [1][1...n] or [1...m][1...n]
A vector (m=1) or matrix with time corresponding to the Temp input variable.
The vector must have a minimum of two entries.  If a matrix argument is
passed it is assumed that each row corresponds to one time-temperature
history.  Time-temperature histories of different length are accomodated
by padding with NaN.  The dimensions of time and temp must agree.
Units are seconds.

temp [1][1...n] or [1...m][1...n]
A vector or matrix with temperatures in degrees C that correspond to the
entries in the time input argument.

E
Activation energy for the reaction in kcal/mol/K

A
Frequency factor (a.k.a. pre-exponential factor) in 1/sec.


INPUT (OPTIONAL):
Iindividual
Integrated reaction rate constant for each reactant at time zero.  Used
to initialize program at F values greater than the minimum.  Each
row corresponds to an activation energy (E) and each column to a
time-temp history.  These values are output by the model so a heating
history prior to this run of the model can be accommodated.

Trxnlim
Minimum temperature below which no reaction is allowed to occur.

OUTPUT:

F, fraction reacted
0-1, Output has same size as input time and Temp.

Findividual
The fraction of each individual component that has reacted.  Each row
corresponds to an activation energy (E) while each column corresponds to
a time and if multiple time-temperature histories are present, the third
matrix dimension corresponds to each history.  The maximum value for any
row is f, the stoichiometry coefficient.

E
Activation energies, kcal/mole

A
Pre-exponential or frequency factor, sec^-1

f
Stoichiometry coefficients (fraction, sums to 1)

Iindividual
Integrated reaction rate constant for each reactant.  Same row x col
structure as Findividual


Version 1.2
GLC modified PJP's script for python

Version 1.1
PJP modified to check for presence of optional inputs without assuming a
certain number of input parameters. Feb. 4, 2014.

Version 1.0
Pratigya J. Polissar (ppolissar@ldeo.columbia.edu)
Modification of easyM2 Version 1.0. That was a modification of EasyR.
Added additional error checking on dimensions of input arguments.
March 6, 2014

EasyR Version 1.0
Pratigya J. Polissar (ppolissar@ldeo.columbia.edu)
November 27, 2012
Modification of easyR Version 2.0:
Fixed bug in setup of DI_init with default zero values.  Would have made
a difference for input matrices where the number of time points was less
than the number of timeseries (would have crashed the program rather than
output erroneous values).

'''

import numpy as np
import pandas as pd
import pint
import pint_pandas
import regex
import sys
from matplotlib import pyplot as plt

def easymolecule_tlim(time, temp, E, A, Trxnlim = False, Iindividual_in = 0):
    
    
    '''R, gas constant kcal/K/mol'''
    Rg = 1.987/1000 
    
    ''' constants used in kinetics during linear temperature increase '''
    a1 = 2.334733 
    a2 = 0.250621
    b1 = 3.330657
    b2 = 1.681534
    
    f = np.array([1])
    
    ''' Check for errors '''
    if type(A) is not float:
        print('A must be a scalar')
    if type(E) is not float:
        print('E must be a scalar') 
    if type(time) is not np.ndarray:
        print('time must be an array')
    if type(temp) is not np.ndarray:
        print('temperature must be an array')
        
    ''' check argument dimensions'''
    if np.shape(time) != np.shape(temp):
        print('dimensions of time and temperature must agree')
        
    if np.size(Iindividual_in) > 1:       
        if np.size(Iindividual_in,0) != np.size(E):
            print('Iindividual_in must have row entries for every E')
        elif np.size(Iindividual_in,1) != np.size(time,0):
            print('Iindividual_in must have columns for every time-temp history')
        
        Findiv_in = np.tile(np.transpose(f),np.shape(Iindividual_in)[0]) * (1 - np.exp(-1 * Iindividual_in))
         
        if (np.tile(np.transpose(f), np.shape(Findiv_in)) - Findiv_in < 0).any():
            print('Individual values give Findividuals that are less than stoichiometry coefficients for that activation energy')
        
            ''' set up initial DI values '''   
    DI_init = Iindividual_in
    if np.size(Iindividual_in) == 1:
        '''if np.shape(time)[0] == np.size(time):
            DI_init = np.zeros([np.size(E),1])
        else:'''
        
        if np.size(E) == 1:
            DI_init = np.zeros((np.size(E),np.size(time,0)))
        else:
            DI_init = np.zeros((np.size(E,0),np.size(time,0)))
        
    ''' Define whether a minimum temperature is necessary for reaction '''

    Tlim = False
    if Trxnlim is not False:
        Tlim = True
        Trxnlim = Trxnlim + 273.15
    
    ''' convert temperature to Kelvin '''

    temp_C = temp 
    temp_K = temp_C + 273.15
    
    ''' Set output matrices '''
    ''' For a single time-temp history '''
    '''if np.size(time) == np.shape(time)[0]:
        F = np.empty([1,np.shape(time)[0]])
        R = np.empty([1,np.shape(time)[0]])
        Iindividual_out = np.empty([np.size(E), np.shape(time)[0], 1])
    else:'''
     
    
    ''' For a multiple time temp histories  '''
    F = np.empty(np.shape(time))
    R = np.empty(np.shape(time))
    
    if np.size(E) == 1:
        Iindividual_out = np.empty([np.size(E), np.size(time,1), np.size(time,0)])
        Findiv = np.empty([np.size(E), np.size(time,1), np.size(time,0)])
    else:
        Iindividual_out = np.empty([np.size(E,0), np.size(time,1), np.size(time,0)])
        Findiv = np.empty([np.size(E,0), np.size(time,1), np.size(time,0)])
        
    
    N_iter = np.arange(np.shape(time)[0])
    
    
    ''' Begin main program loop '''
    ''' for one time/temp history '''
    
    '''if np.size(time,0) == 1: '''
    
    for i in N_iter:
    
        t = time[i,:]
        T = temp_K[i,:]
        
        T = np.tile(T,(1,1))
        t = np.tile(t,(1,1))
        
        ''' Setup heating rate steps (K/s) '''
        H = np.diff(T)/np.diff(t)
        
        ''' Calculate Is '''
        '''creates an E x T matrix'''
        ERT1 = E * (1/(Rg * T)) 
        ERT1 = np.tile(ERT1,[1,1])
        ''' I = np.tile(T, [1,1]) * A * np.exp(-1 * ERT1) * (1 - (ERT1**2 + a1 * ERT1 + a2)/(ERT1**2 + b1 * ERT1 + b2))'''
        I = T * A * np.exp(-1 * ERT1) * (1 - (ERT1**2 + a1 * ERT1 + a2)/(ERT1**2 + b1 * ERT1 + b2))
        
        
        ''' Calcualte individual Is '''
        Hmat = np.tile(H, [np.size(E), 1])
        Hmat0 = (Hmat == 0)
        ''' Set isothermal heating rates to a very small value '''
        Hmat[Hmat0] = 1e-20
        
        difftmat = np.tile(np.diff(t),[1,1])
        DI = (I[0,1:] - I[0,0:-1])/Hmat
        ERT = ERT1[:,0:-1]
        
        ''' Calculate individual DI for isothermal steps'''
        DI[Hmat0] = A * np.exp(-1 * ERT[Hmat0]) * difftmat[Hmat0]
        
        if Tlim is True:
            '''Set all time intervals where either the beginning or ending 
            temperatures are below Trxnlim to zero reaction'''
            Itemp = np.zeros(np.shape(T))
            Itemp[T <= Trxnlim] = 1
            DImatTlim = np.zeros((1,np.shape(T)[1]-1))
            DImatTlim[0,(Itemp[0,0:-1] + Itemp[0,1:]) > 0] = 1
            DImatTlim = np.tile(DImatTlim,(np.shape(DI)[0],1))
            '''DImatTlim_trans = np.transpose(DImatTlim)'''
            DI[DImatTlim > 0] = 0
                
        DI = np.append(DI_init[:,i], DI)
       
        DI = np.tile(DI,(1,1))
    
        test = []
        ''' Calculate a running sum of delta Is ***Focus on here and down''' 
        for t_step in t[0,1:]:
            t_idx = np.where(t[0,:] == t_step)
            t_idx = int(t_idx[0])
            
            test.append(t_idx)
            
            DI[:,t_idx] = DI[:,t_idx] + DI[:,t_idx-1]
            
            
        ''' Populate output matrix'''
        Iindividual_out[:,:,i] = DI
        
        ''' Calculate the cumulative reacted '''
        Findiv[:,:,i] = np.tile(f, (1, np.size(DI,1))) * (1 - np.exp(-1 * DI))
        
        '''Findiv(:,:,n)=repmat(f,1,size(DI,2)).*(1-exp(-DI));'''
        
        ''' Calculate and accumulate results '''
        F[i,:] = sum(Findiv[:,:,i])
        
        
        
    return F, Findiv, E, A, f, Iindividual_out, Tlim



    
    


        
    

    
        
        
        
#%%


''' easymolecule_tlim(np.array([1, 2, 3]), np.array([4,5,6]),5.2,5.1)'''

'''easymolecule_tlim(np.array([0, 3, 5, 4]), np.array([0, 3, 4]), 1.2 , 1.3)

    
 if np.size(Iindividual_in) > 1:
     Findiv_in = np.array([f,]*np.size(Iindividual_in,2)) * (1 - np.exp(-1 * Iindividual_in))
 else:
     Findiv_in = np.array([f,]*np.size(Iindividual_in)) * (1 - np.exp(-1 * Iindividual_in))'''

