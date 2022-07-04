# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 14:52:19 2022

@author: gcoffey
"""

import numpy as np
import pandas as pd
import pint
import pint_pandas
import regex
import sys
from matplotlib import pyplot as plt
from statistics import mean
from scipy import special
from easymolecule import easymolecule30062022 as em


''' Read in biomarker results, results tables should have the following columns:
    Sample name, thermal maturity parameter e.g. MPI4, distance from slip layer) '''
    
class MissingColumnError(Exception):
    """ Something can't be found in the input file """
    
''' filepath configured for pc '''
bio_df = pd.read_csv('/Users/gcoffey/Documents/GitHub/fault-rocks/biomarker_test_MPI4.csv')
bio_cols = pd.Series(bio_df.columns)

''' find the appropriate columns '''
MPI4_pattern = ['MPI4', 'MPI-4', 'MPI3 mod',' MPI-3 mod']

MPI4_esc = [regex.escape(s) for s in MPI4_pattern]
MPI4_pattern = '|'.join(MPI4_esc)
MPI4_cols = [col for col in bio_df.columns if regex.search(MPI4_pattern, col, regex.IGNORECASE)]
MPI4_pick = MPI4_cols[0]

distance_pattern = ['distance ']
distance_esc = [regex.escape(s) for s in distance_pattern]
distance_pattern = '|'.join(distance_esc)
distance_cols = [col for col in bio_df.columns if regex.search(distance_pattern, col, regex.IGNORECASE)]

distance_pick = distance_cols[0]


try:
    MPI4_cols[0]
except IndexError:
    raise MissingColumnError("No thermal maturity columns found")
    
try: 
    distance_cols[0]
except IndexError:
    raise MissingColumnError("No distance columns found") 

''' Define fault and thermal properties '''

''' Fault half width (m, can be a single value or vector of values) '''
width = [5*10**(-3), 1*10**(-2), 1*10**(-3)]
halfwidth = [x/2 for x in width]
N_width = len(halfwidth)
halfwidth_series = pd.Series(halfwidth)

''' Friction (dynamic at this stage, come back and use average friction. This
can be a single value or vector of values) '''
mu = [0.1, 0.6]
N_mu = len(mu)
mu_series = pd.Series(mu)

''' Displacement values to iterate over (vector, m)'''
slip_min = 0.2
slip_max = 3
N_slip = 10
slip = np.linspace(slip_min, slip_max, N_slip)
slip_series = pd.Series(slip)

''' Depth (m) '''
depth = 2000

''' Thermal diffusivity (m2/s) '''
alpha = 1.5 * 10**(-6)

''' slip velocity (m/s, generally should be kept as 1 m/s) '''
v = 0.8

''' Ambient temperature (Celsius) '''
T0 = 300

''' Density (g/m3) '''
density = 2000

''' Heat capacity (J/kgK) '''
cp = 1510

''' Latent heat of fusion '''
H = 0.33 * 10**6


''' Define measurement data '''

''' Distance from slipping layer '''
dist_float = bio_df[distance_pick]

''' Thermal maturity '''
MPI4_float = bio_df[MPI4_pick]

''' Calculate shear stress '''
norm_stress = depth * density * 9.98
Pf = depth * 1000 * 9.98
eff_norm_stress = norm_stress - Pf

tau = [friction * eff_norm_stress for friction in mu]


''' Define a background distance threshold '''
bg_thresh = 0.1
'''Ea = 22.436
A = 12341.0'''

Ea = 19.1714
A = 701.3584
''' Calculate background maturity '''

bg_vect = []

for meas in dist_float:
    if abs(meas) >= 0.1:
        idx = (bio_df[distance_pick] == meas).idxmax()
        bg_vect.append(MPI4_float[idx])

''' Use the average background MPI4 as a bench mark for reaction '''
MPI4_ave_bg = mean(bg_vect)
MPI4_bg = 0.5056
''' Define grid '''
Ng = 50

x = np.logspace(-4, 0.1, Ng+1)
t = np.logspace(-2, 4, Ng)

xmat = np.array([x,]*len(t)).transpose()
tmat = np.array([t,]*len(x))


def i2erfc(z):
    return 1/4 * ((1 + 2*z**2) * special.erfc(z) - 2 * z * np.exp(-1 * z**2)/np.sqrt(np.pi))

t_star_vect = np.zeros(shape = (len(slip), 1));
A0_vect = np.zeros(shape = (len(slip), 1));

for disp in slip:
    disp_idx = (slip_series == disp).idxmax()
    t_star_vect[disp_idx] = slip[disp_idx]/v
    
    
theta = np.zeros(shape=(51, 50, N_slip, N_width, N_mu))
F_output = np.zeros(shape=(51, 50, N_slip, N_width, N_mu))
Tmax_output = np.zeros(shape = (N_slip, N_width, N_mu))
slip_output = np.zeros(shape = (N_slip, N_width, N_mu))
tau_output = np.zeros(shape = (N_slip, N_width, N_mu))
MPI4_output = np.zeros(shape=(51, 50, N_slip, N_width, N_mu))
MPI4max_output = np.zeros(shape = (N_slip, N_width, N_mu))
MPI4_ind_out = np.zeros(shape = (N_slip, N_width, N_mu))
Tmax_ind_out = np.zeros(shape = (N_slip, N_width, N_mu))

for fric in mu:
    for a in halfwidth:
        for disp in slip:
            
            mu_idx = (mu_series == fric).idxmax()
            a_idx = (halfwidth_series == a).idxmax()
            disp_idx = (slip_series == disp).idxmax()
            A0 = (tau[mu_idx] * v) / (2 * halfwidth[a_idx])
            t_star = slip[disp_idx]/v
            
            n1 = (halfwidth[a_idx] - xmat) / np.sqrt(4 * alpha * tmat)
            n2 = (halfwidth[a_idx] + xmat) / np.sqrt(4 * alpha * tmat)
            n3 = (halfwidth[a_idx] - xmat) / np.sqrt(4 * alpha * (tmat - t_star))
            n4 = (halfwidth[a_idx] + xmat) / np.sqrt(4 * alpha * (tmat - t_star))
            n5 = (xmat - halfwidth[a_idx]) / np.sqrt(4 * alpha * tmat)
            n6 = (xmat + halfwidth[a_idx]) / np.sqrt(4 * alpha * tmat)
            n7 = (xmat - halfwidth[a_idx]) / np.sqrt(4 * alpha * (tmat - t_star))
            n8 = (xmat + halfwidth[a_idx]) / np.sqrt(4 * alpha * (tmat - t_star))
            
            n3[np.where(tmat < t_star)] = 0
            n4[np.where(tmat < t_star)] = 0
            n7[np.where(tmat < t_star)] = 0
            n8[np.where(tmat < t_star)] = 0
            
            h = (tmat > t_star)
            h2 = (xmat >= halfwidth[a_idx])
            h3 = (xmat < halfwidth[a_idx])
            
            h = h.astype('int64')
            h2 = h2.astype('int64')
            h3 = h3.astype('int64')
                      
            K1 = i2erfc(n1)
            K2 = i2erfc(n2)
            K3 = i2erfc(n3)
            K4 = i2erfc(n4)
            K5 = i2erfc(n5)
            K6 = i2erfc(n6)
            K7 = i2erfc(n7)
            K8 = i2erfc(n8)
            
            theta[:, :, disp_idx, a_idx, mu_idx] = h3 * (A0 / (density * cp) * (tmat * (1 - 2 * K1 - 2 * K2) - h * (tmat - t_star) * (1 - 2 * K3 - 2 * K4))) + h2 * (A0/(density * cp) * (tmat * (2 * K5 - 2 * K6) - (tmat - t_star) * (2 * K7 - 2 * K8)))
            theta[:, :, disp_idx, a_idx, mu_idx] = theta[:, :, disp_idx, a_idx, mu_idx] + T0
            
            pre_vect = np.linspace(2,10,int((10-2)/0.005+1))
            
            tpre = []
            for num in pre_vect:
                sq = 10**num
                tpre.append(sq)
            tpre = np.array(tpre)  
            tpre = np.tile(tpre,(1,1))
            
            
            ''' changed 200 to T0 as I think this is supposed to burial temp '''
            Tpre = np.tile(T0,np.shape(tpre)[1])
            Tpre = np.tile(Tpre,(1,1))
            
            '''temperature required for reaction'''
            '''Trxnlim1 = 180'''
            
            F, Findiv, E, A, f, Iindividual_bury, Tlim = em.easymolecule_tlim(tpre, Tpre, Ea, A, 20)
            
            MPI4pre = F * 0.85
                       
            ''' Find where pre MPI4 = unheated experiment '''
            ipre1 = MPI4pre > MPI4_bg   
            ipre1 = ipre1.transpose()
            
            if not (ipre1).any() == True:
                print('kinetics burial heating time is not long enough')
                
            '''ipre_all_true = np.where(ipre1 == True)'''
            
            ipre_all_true = [i for i in range(len(ipre1)) if ipre1[i] ==True]
            ipre = ipre_all_true[0]
            
            if (MPI4pre[0,ipre] - MPI4_bg) > 0.05:
                print('kinetics burial time step is too long')
                
            Icalc = np.tile(Iindividual_bury[0,ipre], (1,np.shape(tmat)[0]))
            
            F, Findiv, E, A, f, Iindividual_out, Tlim = em.easymolecule_tlim(tmat, theta[:,:, disp_idx, a_idx, mu_idx ], E, A, 20, Icalc)
            '''  F, Findiv, E, A, f, Iindividual_out, DI, I, H, ERT, Hmat, T, t '''
                
            F_output[:,:,disp_idx, a_idx, mu_idx] = F 
            
            MPI4calc = F * 0.85
            MPI4_output[:,:,disp_idx, a_idx, mu_idx] = MPI4calc
            MPI4max_output[disp_idx, a_idx, mu_idx] = max(MPI4calc[0,:])        
            MPI4max_ind = np.argmax(MPI4calc[0,:], axis = 0)
            MPI4_ind_out[disp_idx, a_idx, mu_idx] = MPI4max_ind
                
            Tmax_output[disp_idx, a_idx, mu_idx] =  np.amax(theta[0, :, disp_idx, a_idx, mu_idx])
            Tmax_ind = np.argmax(theta[0, :, disp_idx, a_idx, mu_idx])
            Tmax_ind_out[disp_idx, a_idx, mu_idx] = Tmax_ind
                       
            slip_output[disp_idx, a_idx, mu_idx] = slip[disp_idx]           
            tau_output[disp_idx, a_idx, mu_idx] = tau[mu_idx]
           

[d1, d2, d3] = np.shape(Tmax_output)           
Tmax_rs = Tmax_output.reshape((d1*d2*d3,1), order = 'F')
tau_rs = tau_output.reshape((d1*d2*d3,1), order = 'F')
slip_rs = slip_output.reshape((d1*d2*d3,1), order = 'F')
MPI4max_rs = MPI4max_output.reshape((d1*d2*d3,1), order = 'F')
Tmax_ind_rs = Tmax_ind_out.reshape((d1*d2*d3,1), order = 'F')
MPI4max_ind_rs = MPI4_ind_out.reshape((d1*d2*d3,1), order = 'F')

theta_rs = theta.reshape(np.shape(theta)[0], np.shape(theta)[1],(d1*d2*d3), order = 'F')
MPI4_out_rs = MPI4_output.reshape(np.shape(theta)[0], np.shape(theta)[1],(d1*d2*d3), order = 'F')

''' Find reacted maturities according to some reaction threshold '''


''' Use the analytical uncertainty (1-sigma) in MPI-4 (this needs to be requantified) '''
sigma = 0.038/2


''' Number of maturities to target '''

MPI4_target = np.array([0.6, 0.62, 0.7])

Tmax_hits = np.zeros(shape = (len(Tmax_rs),len(MPI4_target)))
tau_hits = np.zeros(shape = (len(Tmax_rs),len(MPI4_target)))
slip_hits = np.zeros(shape = (len(Tmax_rs),len(MPI4_target)))
MPI4_hits = np.zeros(shape = (len(Tmax_rs),len(MPI4_target)))
Tmax_ind_hits = np.zeros(shape = (len(Tmax_rs),len(MPI4_target)))
MPI4max_ind_hits = np.zeros(shape = (len(Tmax_rs),len(MPI4_target)))

Tprofile_hits = np.zeros(shape = (len(Tmax_rs), np.shape(theta)[1], len(MPI4_target)))
MPI4profile_hits = np.zeros(shape = (len(Tmax_rs), np.shape(theta)[1], len(MPI4_target))) 


for i in MPI4_target:
    MPI4_idx1 = np.where(MPI4_target == i)
    MPI4_idx = MPI4_idx1[0][0]
    
    hit_idx = np.where(MPI4max_rs[:,0] >= MPI4_target[MPI4_idx] )[0]
    
    idx_range = np.arange(0,len(hit_idx),1)
    
    for idx in idx_range:
       MPI4_hits[idx,MPI4_idx] = MPI4max_rs[hit_idx[idx]]
       tau_hits[idx,MPI4_idx] = tau_rs[hit_idx[idx]]
       slip_hits[idx,MPI4_idx] = slip_rs[hit_idx[idx]]
       Tmax_hits[idx,MPI4_idx] = Tmax_rs[hit_idx[idx]]
       Tmax_ind_hits[idx,MPI4_idx] = Tmax_ind_rs[hit_idx[idx]]
       MPI4max_ind_hits[idx,MPI4_idx] = MPI4max_ind_rs[hit_idx[idx]]
       
       
       Tprofile_hits[idx, :, MPI4_idx] = theta_rs[0, :, hit_idx[idx]]
       

    

                            
            
           



