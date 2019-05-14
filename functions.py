#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 16:08:04 2018

@author: zahramansoor
"""

import numpy as np

#FIXME: clean up syntax in the script - functional but not clean

#%%
def global_to_element_21x21(element_matrix,global_matrix):
    '''
    Converts the elemental nodes to global nodes for the Newton's method
    '''
    
    element_matrix[0]=global_matrix[0] #first node
    j=1
    for i in np.arange(1,1762,4):
        if (i + 3)%84 == 0 and i != 1761: #if the node is on the boundary element, skip two 
                                            #to get to the next global node (boundaries done sep.)
            element_matrix[i] = global_matrix[j] 
            element_matrix[i + 1] = global_matrix[j + 22]
            element_matrix[i + 2] = global_matrix[j + 21]
            element_matrix[i + 3] = global_matrix[j + 1]
            j += 2
        
        elif i == 1761: #since the last element would only have 3 nodes and not the next (the 4th)
            element_matrix[i] = global_matrix[j] 
            element_matrix[i + 1] = global_matrix[j + 22]
            element_matrix[i + 2] = global_matrix[j + 21]
        else:
            element_matrix[i] = global_matrix[j]
            element_matrix[i + 1] = global_matrix[j + 22]
            element_matrix[i + 2] = global_matrix[j + 21]
            element_matrix[i + 3] = global_matrix[j]
            j += 1

    return element_matrix

    
def residual_r(phi, r, n, **params):
    '''
    Calculation of the residual function for the recovery variable r. 
    Takes into account r and phi in each element.
    '''
    
    #set constants
    gamma_AVP = params['gamma_AVP']; mu1 = params['mu1']; mu2 = params['mu2']; diff_t = params['delta_t']
    c_AVP = params['c_AVP']; b_AVP = params['b_AVP']     
    
    R_r = r[n] - r[n-1] - (((gamma_AVP + ((mu1*r[n])/(mu2 + phi))) *
        (-r[n] - ((c_AVP * phi) * (phi - b_AVP - 1)))) * (diff_t))
    
    return R_r


def partial_derivative_r_residual_r(phi, r, **params):
    '''
    Calculation of the partial derivative w/r/t r of the residual function for r.
    Takes into account r and phi in each element.
    '''
    
    #set constants
    gamma_AVP = params['gamma_AVP']
    mu1 = params['mu1']; mu2 = params['mu2']; diff_t = params['delta_t']
    c_AVP = params['c_AVP']; b_AVP = params['b_AVP']     
    
    partial_R_r = 1+(gamma_AVP+(((mu1*r)/(mu2+phi))*((2*r)+((c_AVP*phi)*
                                                            (phi-b_AVP-1)))))*(diff_t)
    return partial_R_r


def d_phi_r(P_r_R_r,P_phi_R_r):
    '''
    Calculates value of d_phi_r which is needed to calculate d_phi_f_phi.
    '''
    d_phi_r = P_phi_R_r/P_r_R_r
    return d_phi_r


def r_FHN(r_n, phi, **params):
    '''
    Value of r in FHN element.
    '''
    #set constants
    a = 0; b = -0.6; diff_t = params['delta_t']
    
    r = r_n +((phi + a)*diff_t)/(1 + b*diff_t)
    
    return r


def partial_derivative_phi_residual_r(phi, r, **params):
    '''
    Calculation of the partial derivative w/r/t phi of the residual function for r.
    Takes into account r and phi in each element.
    '''
    
    #set constants
    gamma_AVP = params['gamma_AVP']; c_AVP = params['c_AVP']; b_AVP = params['b_AVP']
    mu1 = params['mu1']; mu2 = params['mu2']; diff_t = params['delta_t']
    
    partial_phi_R_r = (((gamma_AVP+((mu1*r)/(mu2+phi)))*
            (c_AVP*((2*phi)-b_AVP-1)))-( ((mu1*r) /
            ((mu2+phi)**2))*(r+((c_AVP*phi)*(phi-b_AVP-1)))) )*diff_t
    
    return partial_phi_R_r


def partial_derivative_phi_R_phi_AVP(i, j, phi, r, nodes, dphi_r, n, **params):
    '''
    Calculation of the partial derivative w/r/t phi of residual function.
    Takes into account individual shape functions of each element & the value of phi at each element node.
    
    '''
    
    #initial values
    term3_a = np.zeros((4,4)); term3_b = 0; k = min(nodes)
    #set constants
    c_AVP = params['c_AVP']; diff_t = params['delta_t']; alpha_AVP = params['alpha_AVP']
    nsqint = params['nsqint']; ngradint = params['ngradint']; nquadint = params['nquadint']
    ncubint = params['ncubint']

    #partial derivative term 1; double integral; for phi componenets J = 1, 2, ... 4
    term1 = (1/diff_t) * nsqint[i,j]
    
    #partial derivative term 1; double integral
    term2 = 0.1*ngradint[i,j]
    for k_int in [0,1,2,3]:
        for l in [0,1,2,3]:
            term3_a[k_int,l] = (-3*c_AVP) * ( phi[n,(k_int+k)]*
                                        phi[n,(l+k)] * nquadint[i,j,k_int,l])
        term3_b += ( 2*c_AVP*(1+alpha_AVP) - dphi_r )*( phi[n,(k_int+k)]*
                                                               ncubint[i,j,k_int] )
    term3_c = (alpha_AVP*c_AVP) * ( r[n,(j+k)] * nsqint[i,j])
    term3 = sum(sum(term3_a))+term3_b-term3_c
    
    #sum of individual 'k' nodes
    #sum(sum(..)) sums up all the elements of the matrix
    #partial derivative of residual wrt phi for node - sum of terms
    p_phi_R_phi = term1+term2-term3

    return p_phi_R_phi


def partial_derivative_phi_R_phi_FHN(i,j,phi,r,nodes,n, **params):
    '''
    Calculation of the partial derivative w/r/t phi of residual function for FHN NODES.
    Takes into account individual shape functions of each element & the value of phi at each element node.
    '''
    
    #initial values
    term3_a = np.zeros((4,4)); term3_b = 0; k = min(nodes)
    #set constants
    b_d_phi_f_phi_FHN = params['b_d_phi_f_phi_FHN']; diff_t = params['delta_t']; c_d_phi_f_phi_FHN = params['c_d_phi_f_phi_FHN']
    alpha_d_phi_f_phi_FHN = params['alpha_d_phi_f_phi_FHN']; nsqint = params['nsqint']; ngradint = params['ngradint']; nquadint = params['nquadint']
    ncubint = params['ncubint']; alpha_fun_phi_FHN = params['alpha_fun_phi_FHN']

    #partial derivative term 1; double integral;  for phi componenets J = 1, 2, ... 4
    term1 = (1/diff_t) * nsqint[i,j]
    
    #partial derivative term 1; double integral; defining the constant d_phi_r
    d_phi_r = (c_d_phi_f_phi_FHN*diff_t)/(1 + (b_d_phi_f_phi_FHN*diff_t))
    term2 = 0.1*ngradint[i,j]
    for k_int in [0,1,2,3]:
        for l in [0,1,2,3]:
            term3_a[k_int,l] = (-3*c_d_phi_f_phi_FHN) * ( phi[n,(k_int+k)]*
                                        phi[n,(l+k)] * nquadint[i,j,k_int,l])
        term3_b += 2*c_d_phi_f_phi_FHN*(1+alpha_fun_phi_FHN)*phi[n,(k_int+k)]*ncubint[i,j,k_int]
    term3_c = (alpha_d_phi_f_phi_FHN*c_d_phi_f_phi_FHN + d_phi_r)*( nsqint[i,j] )
    term3 = sum(sum(term3_a))+term3_b-term3_c
    
    #sum of individual 'k' nodes
    #sum(sum(..)) sums up all the elements of the matrix
    #partial derivative of residual wrt phi for node - sum of terms
    p_phi_R_phi = term1+term2-term3

    return p_phi_R_phi


def residual_phi(i, phi, r, nodes, n, **params):
    '''
    Calculation of the residual function.
    Takes into account individual shape functions of each element as well as the phi componenets of each element.
    i and j has to be between 0 and 3 ONLY. Howevever, the matrix index for phi (element node matrix) MUST be j+k, m+k, etc.
    '''
    
    #initial values
    term1=0; term2=0; term3_x=0; term3_y=0; term4_a=0; term4_b=0; term4_c=0; term4_d=0; k = min(nodes)
    
    #set constants
    alpha_fun_phi_AVP = params['alpha_fun_phi_AVP']; c_AVP = params['c_AVP']
    diff_t = params['delta_t']; nsqint = params['nsqint']; ngradint = params['ngradint']; nquadint = params['nquadint']
    ncubint = params['ncubint']; nfluxint = params['nfluxint']
    
    #residual term 1 - double integral
    for j in [0,1,2,3]:
        term1 += (1/diff_t)*( (phi[n,(j+k)]-phi[(n-1),(j+k)])*nsqint[i,j] ) #sum of each 'j' nodes
        
    #residual term 2; double integral
    for j in [0,1,2,3]:
        term2 += (phi[n,(j+k)]*0.1*ngradint[i,j])
    
    #residual term 3; evaluating the flux
    if min(nodes)==0:
        #the 3rd dimension is the edge ; refer to notes for edge #'s
        for j in [0,1,2,3]:
            term3_x += 0.1 * (phi[n,(j+k)] * nfluxint[i,j,2])
            term3_y += 0.1 * (phi[n,(j+k)] * nfluxint[i,j,1])
    elif min(nodes)==80:
        for j in [0,1,2,3]:
            term3_x += 0.1 * (phi[n,(j+k)] * nfluxint[i,j,2])
            term3_y += 0.1 * (phi[n,(j+k)] * nfluxint[i,j,3])
    elif min(nodes)==1680:
        for j in [0,1,2,3]:
            term3_x += 0.1 * (phi[n,(j+k)] * nfluxint[i,j,0])
            term3_y += 0.1 * (phi[n,(j+k)] * nfluxint[i,j,1])
    elif min(nodes)==1760:
        for j in [0,1,2,3]:
            term3_x += 0.1 * (phi[n,(j+k)] * nfluxint[i,j,0])
            term3_y += 0.1 * (phi[n,(j+k)] * nfluxint[i,j,3]) 
    elif min(nodes)%84==0 and min(nodes)!=1680 and min(nodes)!=0: ##left boundary
        for j in [0,1,2,3]:
            term3_x += 0.1 * ( (phi[n,(j+k)] * nfluxint[i,j,0])+
                                 (phi[n,(j+k)] * nfluxint[i,j,2]) )
            term3_y += 0.1 * (phi[n,(j+k)] * nfluxint[i,j,1])
    elif min(nodes)%84==80 and min(nodes)!=1760 and min(nodes)!=80: ##right boundary
        for j in [0,1,2,3]:
            term3_x += 0.1 * ( (phi[n,(j+k)] * nfluxint[i,j,0])+
                                 (phi[n,(j+k)] * nfluxint[i,j,2]) )
            term3_y += 0.1 * (phi[n,(j+k)] * nfluxint[i,j,3])
    elif min(nodes)/83<1 and min(nodes)!=0 and min(nodes)!=80: ##bottom boundary
        for j in [0,1,2,3]:
            term3_x += 0.1 * (phi[n,(j+k)] * nfluxint[i,j,2])
            term3_y += 0.1 * ( (phi[n,(j+k)] * nfluxint[i,j,1])+
                                 (phi[n,(j+k)] * nfluxint[i,j,3]) )
    elif min(nodes)/84>20 and min(nodes)!=1680 and min(nodes)!=1760: ##top boundary
        for j in [0,1,2,3]:
            term3_x += 0.1 * (phi[n,(j+k)] * nfluxint[i,j,0])
            term3_y += 0.1 * ( (phi[n,(j+k)] * nfluxint[i,j,1])+
                                 (phi[n,(j+k)] * nfluxint[i,j,3]) )
    else:
        for j in [0,1,2,3]:
            term3_x += 0.1 * ( (phi[n,(j+k)] * nfluxint[i,j,0])+
                                 (phi[n,(j+k)] * nfluxint[i,j,2]) )
            term3_y += 0.1 * ( (phi[n,(j+k)] * nfluxint[i,j,1])+
                                 (phi[n,(j+k)] * nfluxint[i,j,3]) )
    term3 = term3_x+term3_y #sum of each 'j' nodes and sum of the x-direction
                                       ##and y-direction integrals (area integrals)
    #SOURCE TERM - residual term 4
    #refer to notes to see how the term is distributed
    for j in [0,1,2,3]:
        term4_a += -c_AVP*alpha_fun_phi_AVP*phi[n,(j+k)]*nsqint[i,j]
    for k_it in [0,1,2,3]:
        for l in [0,1,2,3]:
            term4_b += c_AVP*(1+alpha_fun_phi_AVP)*phi[n,(k+k_it)]*phi[n,(l+k)]*ncubint[i,k_it,l]
    for m in [0,1,2,3]:
        for p in [0,1,2,3]:
            for q in [0,1,2,3]:
                term4_c += c_AVP*phi[n,(m+k)]*phi[n,(p+k)]*phi[n,(q+k)]*nquadint[i,m,p,q]
    for v in [0,1,2,3]:
        term4_d += -r[n,v]*phi[n,(v+k)]*nsqint[i,v] #this is different in FHN
    term4 = term4_a+term4_b+term4_c+term4_d
    
    #sum of each 'j' nodes; final term - sum of terms
    R_phi = term1+term2-term3-term4

    return R_phi
                
        
def residual_phi_FHN(i, phi, r, nodes, n, **params):
    '''Calculation of the residual function for FHN NODES.
    Takes into account individual shape functions of each element as well as the phi componenets of each element.
    
    i and j has to be between 0 and 3 ONLY. Howevever, the matrix index for phi (element node matrix) MUST be j+k, m+k, etc.
    '''
    #initial values
    term1=0; term2=0; term3_x=0; term3_y=0; term4_a=0; term4_b=0; term4_c=0; term4_d=0; k = min(nodes)
    
    #set constants
    alpha_fun_phi_FHN = params['alpha_fun_phi_FHN']; c_d_phi_f_phi_FHN = params['c_d_phi_f_phi_FHN']
    diff_t = params['delta_t']; nint = params['nint']; nsqint = params['nsqint']; ngradint = params['ngradint']
    nquadint = params['nquadint']; ncubint = params['ncubint']; nfluxint = params['nfluxint']    
    
    #residual term 1; double integral
    for j in [0,1,2,3]:
        term1 += (1/diff_t)*(phi[n,(j+k)]-phi[(n-1),(j+k)])*nsqint[i,j]
        #sum of each 'j' nodes
        
    #residual term 2; double integral
    for j in [0,1,2,3]:
        term2 += phi[n,(j+k)]*0.1*ngradint[i,j]
        
    #residual term 3; evaluating the flux
    for j in [0,1,2,3]:
        term3_x += 0.1*((phi[n,(j+k)]*nfluxint[i,j,0])+
                                 (phi[n,(j+k)]*nfluxint[i,j,2]))
        term3_y += 0.1*((phi[n,(j+k)]*nfluxint[i,j,1])+
                                 (phi[n,(j+k)]*nfluxint[i,j,3]))
    term3 = term3_x+term3_y #sum of each 'j' nodes and sum of the x-direction
                                       #and y-direction integrals (area integrals)
    #SOURCE TERM - residual term 4 for FHN
    #refer to notes to see how the term is distributed
    
    for j in [0,1,2,3]:
        term4_a += -c_d_phi_f_phi_FHN*alpha_fun_phi_FHN*phi[n,(j+k)]*nsqint[i,j]
    for k_it in [0,1,2,3]:
        for l in [0,1,2,3]:
            term4_b += c_d_phi_f_phi_FHN*(1+alpha_fun_phi_FHN)*phi[n,
                                                (k_it+k)]*phi[n,(l+k)]*ncubint[i,k_it,l]
    for m in [0,1,2,3]:
        for p in [0,1,2,3]:
            for q in [0,1,2,3]:
                term4_c += c_d_phi_f_phi_FHN*phi[n,(m+k)]*phi[n,
                                            (p+k)]*phi[n,(q+k)]*nquadint[i,m,p,q]
    for v in [0,1,2,3]:
        term4_d += -r[n,v]*phi[n,(v+k)]*nint[0,v] #this is different in FHN
    term4 = term4_a+term4_b+term4_c+term4_d
    
    #sum of each 'j' nodes
    #final term - sum of terms
    R_phi = term1+term2-term3-term4
    
    return R_phi

#%%
def element_to_global_21x21(R_phi, p_phi_R_phi):
    '''
    Computes the global matrix of the partial derivatives of the residuals and the residuals and the residuals for the elements.
    '''   
    #initialise
    R_phi_global=np.zeros(484); p_phi_R_phi_global=np.zeros((484,484))
    
    #assembling R_phi's
    R_phi_global[0] = R_phi[0] #standalone nodes
    R_phi_global[21] = R_phi[81]
    R_phi_global[462] = R_phi[1683]
    R_phi_global[483] = R_phi[1762] 
    
    #summing left boundary nodes
    m=3 #the local node start point
    for k in np.arange(22,441,22):
        R_phi_global[k]=R_phi[m]+R_phi[m+81]
        m+=84 #striding by 84
    #summing right boundary nodes
    m=82 #the local node start point
    for k in np.arange(43,463,22):
        R_phi_global[k]=R_phi[m]+R_phi[m+83]
        m+=84
    #summing bottom boundary nodes
    m=1 #the local node start point
    for k in np.arange(1,21,1):
        R_phi_global[k]=R_phi[m]+R_phi[m+3]
        m+=4
    #summing top boundary nodes
    m=1682 #the local node start point
    for k in np.arange(463,483,1):
        R_phi_global[k]=R_phi[m]+R_phi[m+5]
        m+=4
    #summing interior nodes
    k=23 #the global node start point
    m=2 #the local node start point
    while k<461:
        if (k-21)%22==0: #right boundary
            k+=1
        elif k%22==0: #left boundary
            k+=1
            m+=4
        else:
            R_phi_global[k]=R_phi[m]+R_phi[m+5]+R_phi[m+83]+R_phi[m+86] #interior nodes
            k+=1
            m+=4
            
    #assembling p_phi_R_phi's; global_phi (the derivative variable)= column, R = row
    #summing corners
    p_phi_R_phi_global[0,0] = p_phi_R_phi[0,0]
    p_phi_R_phi_global[0,1] = p_phi_R_phi[0,1]
    p_phi_R_phi_global[0,22] = p_phi_R_phi[0,3]
    p_phi_R_phi_global[0,23] = p_phi_R_phi[0,2]

    p_phi_R_phi_global[21,20] = p_phi_R_phi[81,80]
    p_phi_R_phi_global[21,21] =  p_phi_R_phi[81,81]
    p_phi_R_phi_global[21,42] =  p_phi_R_phi[81,83]
    p_phi_R_phi_global[21,43] =  p_phi_R_phi[81,82]

    p_phi_R_phi_global[462,440] =  p_phi_R_phi[1683,1680]
    p_phi_R_phi_global[462,441] =  p_phi_R_phi[1683,1681]
    p_phi_R_phi_global[462,462] =  p_phi_R_phi[1683,1683]
    p_phi_R_phi_global[462,463] =  p_phi_R_phi[1683,1682]

    p_phi_R_phi_global[483,460] =  p_phi_R_phi[1762,1760]
    p_phi_R_phi_global[483,461] =  p_phi_R_phi[1762,1661]
    p_phi_R_phi_global[483,482] =  p_phi_R_phi[1762,1763]
    p_phi_R_phi_global[483,483] =  p_phi_R_phi[1762,1762]
    #summing left boundary
    m=0 #phi that is being using to take the derivative w/r/t
    p=3 #position adjustment (element)
    for k in np.arange(22,441,22):
        p_phi_R_phi_global[k,m] = p_phi_R_phi[p,(p-3)]
        p_phi_R_phi_global[k,(m+1)] = p_phi_R_phi[p,(p-2)]
        p_phi_R_phi_global[k,(m+22)] = p_phi_R_phi[p,p]+p_phi_R_phi[(p+81),(p+81)]
        p_phi_R_phi_global[k,(m+23)] = p_phi_R_phi[(p+81),(p+82)]+p_phi_R_phi[p,(p-1)]
        p_phi_R_phi_global[k,(m+44)] = p_phi_R_phi[(p+81),(p+84)]
        p_phi_R_phi_global[k,(m+45)] = p_phi_R_phi[(p+81),(p+83)]
        m+=22
        p+=84
    #summing right boundary
    m=20 #phi that is being using to take the derivative w/r/t
    p=82 #position adjustment (element)
    for k in np.arange(43,462,22):
        p_phi_R_phi_global[k,m] = p_phi_R_phi[p,(p-2)]
        p_phi_R_phi_global[k,(m+1)] = p_phi_R_phi[p,(p-1)]
        p_phi_R_phi_global[k,(m+22)] = p_phi_R_phi[p,(p+1)]+p_phi_R_phi[(p+83),(p+82)]
        p_phi_R_phi_global[k,(m+23)] = p_phi_R_phi[p,p]+p_phi_R_phi[(p+83),(p+83)]
        p_phi_R_phi_global[k,(m+44)] = p_phi_R_phi[(p+83),(p+84)]
        p_phi_R_phi_global[k,(m+45)] = p_phi_R_phi[(p+83),(p+85)]
        m+=22
        p+=84
    #summing bottom boundary
    m=0 #phi that is being using to take the derivative w/r/t
    p=1 #position adjustment (element)
    for k in np.arange(1,21,1):
        p_phi_R_phi_global[k,m] = p_phi_R_phi[p,(p-1)]
        p_phi_R_phi_global[k,(m+1)] = p_phi_R_phi[p,p]+p_phi_R_phi[(p+3),(p+3)]
        p_phi_R_phi_global[k,(m+2)] = p_phi_R_phi[(p+3),(p+4)]
        p_phi_R_phi_global[k,(m+22)] = p_phi_R_phi[p,(p+2)]
        p_phi_R_phi_global[k,(m+23)] = p_phi_R_phi[p,(p+1)]+p_phi_R_phi[(p+3),(p+6)]
        p_phi_R_phi_global[k,(m+24)] = p_phi_R_phi[(p+3),(p+5)]
        m+=1
        p+=4
    #summing top boundary
    m=440 #phi that is being using to take the derivative w/r/t
    p=1682 #position adjustment (element)
    for k in np.arange(463,483,1):
        p_phi_R_phi_global[k,m] = p_phi_R_phi[p,(p-2)]
        p_phi_R_phi_global[k,(m+1)] = p_phi_R_phi[p,(p-1)]+p_phi_R_phi[(p+5),(p+2)]
        p_phi_R_phi_global[k,(m+2)] = p_phi_R_phi[(p+5),(p+3)]
        p_phi_R_phi_global[k,(m+22)] = p_phi_R_phi[p,(p+1)]
        p_phi_R_phi_global[k,(m+23)] = p_phi_R_phi[p,p]+p_phi_R_phi[(p+5),(p+5)]
        p_phi_R_phi_global[k,(m+24)] = p_phi_R_phi[(p+5),(p+4)]
        m+=1
        p+=4
    #summing interior nodes
    k=23 #global node
    m=0 #phi that is being using to take the derivative w/r/t
    p=2 #position adjustment (element)
    while k<461:
        if k%22==0:
            k+=1
            m+=1 #have to do this step otherwise it shifts the element one over 
        elif (k-21)%22==0:
            k+=1
            m+=1
            p+=4
        else: 
            p_phi_R_phi_global[k,m] = p_phi_R_phi[p,(p-2)]
            p_phi_R_phi_global[k,(m+1)] = p_phi_R_phi[p,(p-1)]+p_phi_R_phi[(p+5),
                                                                        (p+2)]
            p_phi_R_phi_global[k,(m+2)] = p_phi_R_phi[(p+5),(p+3)]
            p_phi_R_phi_global[k,(m+22)] = p_phi_R_phi[p,(p+1)]+p_phi_R_phi[(p+83),
                                                                            (p+82)]
            p_phi_R_phi_global[k,(m+23)] = p_phi_R_phi[p,p]+p_phi_R_phi[(p+83),
                            (p+83)]+p_phi_R_phi[(p+5),(p+5)]+p_phi_R_phi[(p+86),(p+86)]
            p_phi_R_phi_global[k,(m+24)] = p_phi_R_phi[(p+86),
                                                (p+87)]+p_phi_R_phi[(p+5),(p+4)]
            p_phi_R_phi_global[k,(m+44)] = p_phi_R_phi[(p+83),(p+85)]
            p_phi_R_phi_global[k,(m+45)] = p_phi_R_phi[(p+83),
                                                (p+84)]+p_phi_R_phi[(p+86),(p+89)]
            p_phi_R_phi_global[k,(m+46)] = p_phi_R_phi[(p+86),(p+88)]
            k+=1
            m+=1
            p+=4

    return R_phi_global, p_phi_R_phi_global
                                                