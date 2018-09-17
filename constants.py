##FEM computation of a square element of cardiac muscle tissue
##Zahra Dhanerawala, Viktor Grigoryan
##May 27, 2018
##Constants file
import numpy as np

matrix_size = 21*21*4
global_matrix_size = 22*22

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
R_r_step = np.zeros(matrix_size);
t = np.arange(0,0.01,0.001) ##the third value is the desired delta t
diff_t = t[2]-t[1] ##delta t

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
