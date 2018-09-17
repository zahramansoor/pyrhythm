##FEM computation of a square element of cardiac muscle tissue
##Zahra Dhanerawala, Viktor Grigoryan
##May 27, 2018
##Initialisation file

import numpy as np

##define matrix size
matrix_size = 21*21*4
global_matrix_size = 22*22
time_step = 100
##phi is defined for each element according to its shape functions as
##phi = phi_1 .* N_1 + phi_2 .* N_2 + phi_3 .* N_3 + phi_4 .* N_4

##initial guess for phi
##each element in the phi vector corresponds to a node in the system
phi = np.zeros((time_step,matrix_size))

##initial guess for r
##each element in the r vector corresponds to a node in the system
r = np.zeros((time_step,matrix_size))

##---------------------------------------------------------------------------

##per element level calculation!
##intiliase
R_phi = np.ones((matrix_size,1))
p_phi_R_phi = np.zeros((matrix_size,matrix_size))
p_phi_R_phi_global = np.zeros((global_matrix_size,global_matrix_size))
p_r_R_r = np.zeros((time_step,matrix_size))
phi_global_time = np.zeros((global_matrix_size,time_step))
##defined this way so it makes organising the matrices easier
phi_step = np.zeros(time_step)
dphi_r = np.zeros((1,global_matrix_size))
##this vector consists of 10 times steps
t = np.arange(0,0.01,0.001) ##the third value is the desired delta t
diff_t = t[2]-t[1] ##delta t
time = t/diff_t+1 ##where n = 1, 2, ..., 100
time = time.astype(int) ##converts time to integer values

##put initial guess for phi (FHN is non zero)
phi_global = np.zeros(global_matrix_size)
phi_global[230] = 5.385e-1
phi_global[231] = 5.385e-1
phi_global[252] = 5.385e-1
phi_global[253] = 5.385e-1

##setting t=0
import functions as f
phi[0,:]=f.global_to_element_21x21(phi[0,:],phi_global)
print('These are the FHN non-zero nodes' + '\n' +
      str(np.nonzero(phi[0,:]>0))) ##find out non-zero indices of phi

##defining constants
alpha_AVP = 0.01
c_AVP = 8
alpha_d_phi_f_phi_FHN = 0.5
c_d_phi_f_phi_FHN = 50
b_d_phi_f_phi_FHN = -0.6
alpha_fun_phi_AVP = 0.01
alpha_fun_phi_FHN = 0.5
gamma_AVP = 0.002
b_AVP = 0.15
mu1 = 0.2
mu2 = 0.3
a_r_FHN = 0
b_r_FHN = -0.6
tol = 1e-6 ##convergence tolerance
R_r_step = np.zeros((matrix_size, 1));

##shape functions
##importing matlab .mat files for integral values of shape functions
import scipy.io as sio
nint_contents = sio.loadmat('nint.mat')
nint=nint_contents['nint']
nsqint_contents = sio.loadmat('nsqint.mat') ##2D
nsqint=nsqint_contents['nsqint']
ncubint_contents = sio.loadmat('ncubint.mat') ##3D
ncubint=ncubint_contents['ncubint']
nquadint_contents = sio.loadmat('nquadint.mat') ##4D
nquadint=nquadint_contents['nquadint']
ngradint_contents = sio.loadmat('ngradint.mat') ##2D
ngradint=ngradint_contents['ngradint']
nfluxint_contents = sio.loadmat('nfluxint.mat') ##3D
nfluxint=nfluxint_contents['nfluxint']


