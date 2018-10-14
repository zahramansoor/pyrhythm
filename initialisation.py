#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 10:36:52 2018

@author: zahramansoor

"""
##FEM computation of a square element of cardiac muscle tissue
##Initialisation file

from __future__ import division
import numpy as np, scipy.io as sio
from functions import global_to_element_21x21

def fill_params(time_step, matrix_size, global_matrix_size):
    '''Function to initialise cardiac action potential model based on finite element method outlined in Goktepe et al.
    
    Inputs:
        time_step = time to iterate through
        matrix_size = matrix to imitate heart tissue. largest functionality is currently 64x64.
        
    Returns:
        paramter dictionary with initial values and constants
        
    phi is defined for each element according to its shape functions as:
        phi = phi_1 .* N_1 + phi_2 .* N_2 + phi_3 .* N_3 + phi_4 .* N_4
    '''
    params = {}
    
    #initial guess for phi - each element in the phi vector corresponds to a node in the system
    phi = np.zeros((time_step, matrix_size))
    
    #initial guess for r - each element in the r vector corresponds to a node in the system
    r = np.zeros((time_step,matrix_size))
    params['r'] = r 
    
    #per element level calculation
    params['R_phi'] = np.ones(matrix_size)
    params['R_phi_global'] = np.ones(global_matrix_size) #this acts as a dummy variable to start the while loop
    params['p_phi_R_phi'] = np.zeros((matrix_size, matrix_size))
    params['p_phi_R_phi_global'] = np.zeros((global_matrix_size, global_matrix_size))
    params['p_r_R_r'] = np.zeros((time_step,matrix_size))
    params['R_r_step'] = np.zeros(matrix_size) 
    
    params['phi_global_time'] = np.zeros((global_matrix_size, time_step))
    
    #defined this way so it makes organising the matrices easier
    params['phi_step'] = np.zeros(time_step)
    params['dphi_r'] = np.zeros(global_matrix_size)
    
    #put initial guess for phi (FHN is non zero)
    phi_global = np.zeros(global_matrix_size)
    phi_global[230] = 5.385e-1; phi_global[231] = 5.385e-1; phi_global[252] = 5.385e-1; phi_global[253] = 5.385e-1
    params['phi_global'] = phi_global
    
    #update phi_global_time initial values at t = 0
    params['phi_global_time'][:, 0] = phi_global
    
    #setting t=0
    phi[0, :] = global_to_element_21x21(phi[0, :],phi_global)
    params['phi'] = phi

    return params

def fill_constants(time_step):
    
    constants = {}
    
    #defining constants
    constants['alpha_AVP'] = 0.01
    constants['c_AVP'] = 8
    constants['alpha_d_phi_f_phi_FHN'] = 0.5
    constants['c_d_phi_f_phi_FHN'] = 50
    constants['b_d_phi_f_phi_FHN'] = -0.6
    constants['alpha_fun_phi_AVP'] = 0.01
    constants['alpha_fun_phi_FHN'] = 0.5
    constants['gamma_AVP'] = 0.002
    constants['b_AVP'] = 0.15
    constants['mu1'] = 0.2
    constants['mu2'] = 0.3
    constants['a_r_FHN'] = 0
    constants['b_r_FHN'] = -0.6
    constants['tol'] = 1e-6 ##convergence tolerance
    
    # vector consists of 10 times steps
    #FIXME: fix the range issue for different time steps
    t = np.arange(0, time_step/1000, 0.001) #the third value is the desired delta t
    constants['delta_t'] = 0.001
    time = t/constants['delta_t'] #where n = 1, 2, ..., 100
    time = time.astype(int) 
    constants['time'] = time
    
    #shape functions
    nint_contents = sio.loadmat('nint.mat')
    constants['nint'] = nint_contents['nint']
    nsqint_contents = sio.loadmat('nsqint.mat') ##2D
    constants['nsqint'] = nsqint_contents['nsqint']
    ncubint_contents = sio.loadmat('ncubint.mat') ##3D
    constants['ncubint'] = ncubint_contents['ncubint']
    nquadint_contents = sio.loadmat('nquadint.mat') ##4D
    constants['nquadint'] = nquadint_contents['nquadint']
    ngradint_contents = sio.loadmat('ngradint.mat') ##2D
    constants['ngradint'] = ngradint_contents['ngradint']
    nfluxint_contents = sio.loadmat('nfluxint.mat') ##3D
    constants['nfluxint'] = nfluxint_contents['nfluxint']
    
    return constants
        
#%%
if __name__ == '__main__':
    
    time_step = 100
    matrix_size = 21*21*4
    global_matrix_size = 22*22
    
    params = fill_params(time_step, matrix_size, global_matrix_size)
    constants = fill_constants(time_step)