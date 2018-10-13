#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 16:13:52 2018

@author: zahramansoor

"""
import numpy as np
from functions import residual_r, partial_derivative_r_residual_r, partial_derivative_r_residual_r, partial_derivative_phi_residual_r, d_phi_r, partial_derivative_phi_R_phi_AVP, residual_phi

def element(nodes, n, **params):
    '''
    Function makes a loop to calculate residuals and partial derivatives of the 1764 element nodes.
    THIS IS SPECIFIC FOR ALIEV-PANFILOV CELLS
    
    Inputs: 
        nodes = elemental nodes of the position in the matrix
        n = time step
        params = parameter dictionary
    '''
    
    #set matrices from parameter dictionary
    params['phi'] = phi; params['r'] = r; params['R_phi'] = R_phi; params['p_phi_R_phi'] = p_phi_R_phi
    
    k = min(nodes)
    for i in np.arange(0, 3): ##need to do this otherwise the index will be -
        R_r = 1
        while R_r > tol: ##update r using Newton's method
            R_r_step[i+k] += 1
            R_r = residual_r(phi[n,(i+k)], r[:,(i+k)], n)
            p_r_R_r = partial_derivative_r_residual_r(phi[n,(i+k)],
                                                               r[n,(i+k)])
            r[n,(i+k)] = r[n,(i+k)]-(R_r/p_r_R_r)

        ##parameters needed for the partial derivative of phi w/r/t R_phi
        for j in [0,1,2,3]:
            ##these are all computed at the node level
            P_r_R_r = partial_derivative_r_residual_r(phi[n,(i+k)],
                                                      r[n,(i+k)])
            P_phi_R_r = partial_derivative_phi_residual_r(phi[n,(i+k)],
                                                      r[n,(i+k)])
            dphi_r = d_phi_r(P_r_R_r,P_phi_R_r)
            p_phi_R_phi[(j+k),(i+k)] = partial_derivative_phi_R_phi_AVP(i, j,
                                                phi, r, nodes, dphi_r, n)
        #calling the residual function
        R_phi[i+k] = residual_phi(i, phi, r, nodes, n)
    
    #feeds matrices back into the dict
    #FIXME: make function to do this
    phi = params['phi']; r = params['r']; R_phi = params['R_phi']; p_phi_R_phi = params['p_phi_R_phi']
    
    return params


