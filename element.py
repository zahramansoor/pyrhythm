##This function makes a loop to calculate residuals and partial derivatives of the
##1764 element nodes
##THIS IS SPECIFIC FOR ALIEV-PANFILOV CELLS
from constants import *
import functions as f ##importing functions file

def element_calcs(nodes, R_phi, p_phi_R_phi, phi, r, n):
    k=min(nodes)
    for i in [0,1,2,3]: ##need to do this otherwise the index will be -
        R_r=1
        while R_r>tol: ##update r using Newton's method
            R_r_step[i+k]+=1
            R_r = f.residual_r(phi[n,(i+k)], r[:,(i+k)], n)
            p_r_R_r = f.partial_derivative_r_residual_r(phi[n,(i+k)],
                                                               r[n,(i+k)])
            r[n,(i+k)] = r[n,(i+k)]-(R_r/p_r_R_r)

        ##parameters needed for the partial derivative of phi w/r/t R_phi
        for j in [0,1,2,3]:
            ##these are all computed at the node level
            P_r_R_r = f.partial_derivative_r_residual_r(phi[n,(i+k)],
                                                      r[n,(i+k)])
            P_phi_R_r = f.partial_derivative_phi_residual_r(phi[n,(i+k)],
                                                      r[n,(i+k)])
            dphi_r = f.d_phi_r(P_r_R_r,P_phi_R_r)
            p_phi_R_phi[(j+k),(i+k)] = f.partial_derivative_phi_R_phi_AVP(i, j,
                                                phi, r, nodes, dphi_r, n)
        ##calling the residual function
        R_phi[i+k] = f.residual_phi(i, phi, r, nodes, n)

    return (r,R_phi,p_phi_R_phi)


