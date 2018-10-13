#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 10:36:52 2018

@author: zahramansoor

"""
################This version treats only the element nodes in the FHN element as FHN nodes################

import sys, numpy as np
import matplotlib.pyplot as plt
from element import element 
from functions import global_to_element_21x21, r_FHN, residual_phi_FHN, partial_derivative_phi_R_phi_FHN, element_to_global_21x21
from initialisation import fill_params

#define matrix size
matrix_size = 21*21*4
global_matrix_size = 22*22
time_step = 10

#initialise
params = fill_params(time_step, matrix_size, global_matrix_size)

#%%
#time loop
for n in params['time']:
    
    #initalise things
    if n != 0:
        phi_global = params['phi_global_time'][:, (n - 1)] #using the 'passed' phi_global frm previous time
    else:
        phil_global = params['phi_global']
        params['r'][n, :] = params['r'][(n - 1), :]
        
    R_phi_global = params['R_phi_global'];  Tol = params['R_phi_global']; phi_step = params['phi_step']
    phi = params['phi']
    
    sys.stdout.write('\n\n     Starting iterations for n = {}\n\n'.format(n)); sys.stdout.flush()
    
    while sum(abs(R_phi_global)) > sum(params['Tol']):
                
        phi_step[n] += 1
        
        sys.stdout.write('\n\n     Redistributing the phi_global the elements for element-level calculations\n\n'); sys.stdout.flush()

        params['phi'][n,:] = global_to_element_21x21(np.zeros(1764), phi_global)
        
        for z_d in np.arange(0, 1761, 4):
            nodes = [z_d, (z_d + 1), (z_d + 2), (z_d + 3)] ##assign node values & print variable
            params = element(nodes, n, **params)
            
            sys.stdout.write('\n     Nodes: {}\n\n    Residuals: {}\n\n'.format(nodes, params['R_phi'][z_d:(z_d+4)])); sys.stdout.flush()
            
            #FHN element
            if z_d == 880:
                nodes = [z_d, (z_d + 1), (z_d + 2), (z_d + 3)]
                k = min(nodes)
                
                #initalise things
                r = params['r']; R_phi = params['R_phi']; phi = params['phi']; p_phi_R_phi = params['p_phi_R_phi']
                
                for j in [0,1,2,3]:
                    r[n, (j+k)] = r_FHN(r[(n - 1), (j+k)], phi[n,(j+k)])
                    R_phi[j+k] = residual_phi_FHN(j, phi, r, nodes, n)
                    
                    #partial derivative matrix
                    for m in [0,1,2,3]:
                        p_phi_R_phi[(m+k), (j+k)] = partial_derivative_phi_R_phi_FHN(j, m, phi, r, nodes, n)
            
            params['p_phi_R_phi'] = p_phi_R_phi; params['phi'] =phi; params['R_phi'] = R_phi; params['r'] = r
                
            sys.stdout.write('\n     Nodes: {}\n\n    Residuals: {}\n\n'.format(nodes, params['R_phi'][z_d:(z_d + 4)])); sys.stdout.flush()

        sys.stdout.write('\n\n     Assembling 1764 element nodes to 484 global nodes...\n\n'); sys.stdout.flush()
        
        R_phi_global, p_phi_R_phi_global = element_to_global_21x21(R_phi, p_phi_R_phi)
        
        #Newton's iteration for phi
        phi_global = phi_global - np.matmul(np.linalg.inv(p_phi_R_phi_global), R_phi_global)
        
        sys.stdout.write('\n\n     Maximum value of residual for iter {}: {}'.format(n, max(abs(R_phi_global))))

        params['phi_global_time'][:,n] = phi_global #updates global phi matrix for time loop
    
    sys.stdout.write('\n\nPassed tol for n = {}'.format(n))


#%%
##=========================================================================================================    
##visualises the phi_global values by converting the global nodal values
##into a square matrix and generating a heat map
what_time=int(input('Which time step would you like to visualize (only input integers)? ')) ##input integer
while what_time>=1 and what_time<10:
    phi_global_n=phi_global_time[:,what_time] 
    zd=np.zeros((22,22)) ##zd is 22x22 matrix which will be plotted as a heatmap
    k=0
    for i in np.arange(0,483,22):
        zd[k,0:21]=phi_global_n[i:(i+21)]
        k+=1
    plt.imshow(zd, cmap='hot', interpolation='nearest') ##plots heat map
    plt.show() ##shows heat map
    what_time=int(input('Which time step would you like to visualize (only input integers)? ')) ##input integer

              
        
                
                                   

            
                                   
              
