#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 10:36:52 2018

@author: zahramansoor

"""
##This version treats only the element nodes in the FHN element as FHN nodes
import numpy as np
import matplotlib.pyplot as plt
import element_calcs ##imports element_calcs functions
import functions as f

##define matrix size
matrix_size = 21*21*4
global_matrix_size = 22*22
time_step = 100

#initialise
params = f.initialise_params(time_step, matrix_size, global_matrix_size)

##time loop
for n in params['time']:
    R_phi_global = np.ones((global_matrix_size,1))
    ##this acts as a dummy variable to start the while loop
    ##the variable is updated at the end of the while loop to the values of R_phi_global
    Tol = np.ones((global_matrix_size,1))*tol
    phi_global = phi_global_time[:,(n-1)] ##using the 'passed' phi_global frm previous time

    while sum(abs(R_phi_global)) > sum(Tol):
        print('The residuals have not passed tolerance. Recalculating...')
        phi_step[n]+=1
        ##redistributing the phi_global from the %%previous iteration%% or the
        ##initial one defined in the initialisation file to the elements
        ##for element-level calculations
        phi[n,:] = f.global_to_element_21x21(np.zeros(1764),phi_global)
        for z_d in np.arange(0,1761,4):
            nodes = [z_d,(z_d+1),(z_d+2),(z_d+3)] ##assign node values & print variable
            (r,R_phi,p_phi_R_phi)=element_calcs(nodes,R_phi,p_phi_R_phi,phi,r,n)
            print('These are the residual values for the current element with nodes '+
                  str(nodes) + ':'+'\n' +
                  str(R_phi[z_d:(z_d+4)])+'.' )
            ##FHN element
            if z_d==880:
                nodes=[z_d,(z_d+1),(z_d+2),(z_d+3)]
                k=min(nodes)
                for j in [0,1,2,3]:
                    r[n,(j+k)] = f.r_FHN(r[(n-1),(j+k)],phi[n,(j+k)])
                    R_phi[j+k] = f.residual_phi_FHN(j,phi,r,nodes,n)
                    for m in [0,1,2,3]:  ##partial derivative matrix
                        p_phi_R_phi[(m+k),(j+k)] = f.partial_derivative_phi_R_phi_FHN(j,m,phi,r,nodes,n)
                print('These are the residual values for the current element with FHN nodes '+
                      str(nodes) + ':'+'\n' +
                      str(R_phi[z_d:(z_d+4)])+'.')
        ##assembling 1764 element nodes to 484 global nodes
        (R_phi_global,p_phi_R_phi_global)=f.element_to_global_21x21(R_phi, p_phi_R_phi)
        ##Newton's iteration for phi
        phi_global=phi_global-np.matmul(np.linalg.inv(p_phi_R_phi_global),R_phi_global)
        print('This is the maximum value of the residual for the current iteration: '
              + str(max(abs(R_phi_global)))+'.')
        phi_global.shape=(global_matrix_size,) ##makes shape compatible for matrix update
        phi_global_time[:,n]=phi_global ##updates global phi matrix for time loop
    
    print('Congrats! The algotrithm has finished a time loop & passed tolerance for the '
           + str(n) + 'th time step.')
    if n!=9: ##set the guess for the next time step as the solution for the previous time step
        r[(n+1),:]=r[n,:]

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

              
        
                
                                   

            
                                   
              
