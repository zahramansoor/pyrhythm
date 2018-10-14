#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 10:36:52 2018

@author: zahramansoor

"""
################This version treats only the element nodes in the FHN element as FHN nodes################

import sys, numpy as np
from element import element 
from functions import global_to_element_21x21, r_FHN, residual_phi_FHN, partial_derivative_phi_R_phi_FHN, element_to_global_21x21
from initialisation import fill_params, fill_constants
from skimage.external import tifffile

def run_rhythm(params, constants, verbose = False):
    
    for n in constants['time'][1:]:
    
        #initalise/update things
        if n != 0:
            phi_global = params['phi_global_time'][:, (n - 1)] #using the 'passed' phi_global frm previous time
        else:
            phi_global = params['phi_global']
                                  
        sys.stdout.write('\n\n***************************************************************     Starting iterations for n = {}\n\n'.format(n)); sys.stdout.flush()
        
        Tol = np.ones((global_matrix_size, 1))*constants['tol'] #using global matrix size
        
        #re-initialise R_phi_global to start while loop
        params['R_phi_global'] = np.ones(global_matrix_size)
        
        while sum(abs(params['R_phi_global'])) > sum(Tol):
                    
            params['phi_step'][n] += 1
            
            if verbose: sys.stdout.write('\n     Redistributing the phi_global the elements for element-level calculations\n\n'); sys.stdout.flush()
            
            #redistribute/initialise things
            params['phi'][n,:] = global_to_element_21x21(np.zeros(1764), phi_global)
            params['r'][n, :] = params['r'][(n - 1), :]
        
            for z_d in np.arange(0, 1761, 4):
                nodes = [z_d, (z_d + 1), (z_d + 2), (z_d + 3)] ##assign node values & print variable
                params = element(nodes, n, params, constants)                
                if verbose: sys.stdout.write('\n     Nodes: {}\n\n    Residuals: {}\n\n'.format(nodes, params['R_phi'][z_d:(z_d+4)])); sys.stdout.flush()
                
                #FHN element
                if z_d == 880:
                    nodes = [z_d, (z_d + 1), (z_d + 2), (z_d + 3)]
                    k = min(nodes)
                    
                    #initalise things to feed into functions
                    r = params['r']; phi = params['phi']; 
                    
                    for j in [0,1,2,3]:
                        params['r'][n, (j + k)] = r_FHN(r[(n - 1), (j + k)], phi[n, (j + k)], **constants)
                        params['R_phi'][j + k] = residual_phi_FHN(j, phi, r, nodes, n, **constants)
                        
                        #partial derivative matrix
                        for m in [0,1,2,3]:
                            params['p_phi_R_phi'][(m + k), (j + k)] = partial_derivative_phi_R_phi_FHN(j, m, phi, r, nodes, n, **constants)
                                                
                if verbose: sys.stdout.write('\n     Nodes: {}\n\n    Residuals: {}\n\n'.format(nodes, params['R_phi'][z_d:(z_d + 4)])); sys.stdout.flush()
    
            if verbose: sys.stdout.write('\n\n     Assembling 1764 element nodes to 484 global nodes...\n\n'); sys.stdout.flush()
            
            params['R_phi_global'], params['p_phi_R_phi_global'] = element_to_global_21x21(params['R_phi'], params['p_phi_R_phi'])
            
            #Newton's iteration for phi
            phi_global = phi_global - np.matmul(np.linalg.inv(params['p_phi_R_phi_global']), params['R_phi_global'])
            
            sys.stdout.write('\n\n     Maximum value of residual for n = {}: {}\n'.format(n, max(abs(params['R_phi_global'])))); sys.stdout.flush()
    
            params['phi_global_time'][:, n] = phi_global #updates global phi matrix for time loop
        
        sys.stdout.write('\n\nPassed tol for n = {}'.format(n)); sys.stdout.flush()
    
    return params

def viewer(constants, data_dict):
    '''Function to take an array of phi values at particular time steps and output a tif file.
    Tif file will have phi values accross the square tissue matrix for time steps 
    In tif file, z --> time; x, y --> 2D borders of 'cell' in square tissue matrix
    
    Input:
        constants dict
        data_dict = output of run_rhythm
    '''
    arr = np.zeros((len(constants['time']), 22,22)) #22x22 matrix which will be plotted as a heatmap
    
    #making the action potential array into a tif file
    for t in range(len(constants['time'])):
        phi_global_n = data_dict['phi_global_time'][:, t] 
        k = 0
        for i in np.arange(0, 483, 22):
            arr[t, k, 0:21] = phi_global_n[i:(i + 21)]
            k += 1
    
    tifffile.imsave('test.tif', arr.astype('float32'))
    
#%%   
if __name__ == '__main__':
    
    #define matrix size
    matrix_size = 21*21*4
    global_matrix_size = 22*22
    time_step = 100
    
    #initialise
    params = fill_params(time_step, matrix_size, global_matrix_size)
    constants = fill_constants(time_step)
    
    #run algorithm
    data_dict = run_rhythm(params, constants, verbose = False)
    
    #save out tif of output
    viewer(constants, data_dict)
