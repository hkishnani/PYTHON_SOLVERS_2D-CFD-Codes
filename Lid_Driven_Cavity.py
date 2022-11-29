# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 17:29:37 2022

@author: mayan
"""
#%% Libraries
import numpy as np
from scipy.sparse.linalg import bicgstab
from scipy.sparse import spdiags
import matplotlib.pyplot as plt
from numba import njit
#%% Initialization

bicgstab_flag = 0

# Gamma
Re = 1000.0
U_lid = 1.0

# Domain
L_x = 1.0
L_y = 1.0

# Collocated Grid points
N_x = 256
N_y = 256

# Number of rows and cols
nrows = N_y  # Taking no Ghosts
mcols = N_x  # Taking no Ghosts

# Mesh size
dx = L_x/N_x
dy = L_y/N_y

# Collocated mesh points
x_coll = np.linspace(start = dx/2, stop = L_x-dx/2, num = N_x)
y_coll = np.linspace(start = dy/2, stop = L_y-dy/2, num = N_y)

# Primitive variables (stored on mesh points)
u, u_star = np.zeros((2,nrows,mcols), dtype = np.float64)
v, v_star = np.zeros((2,nrows,mcols), dtype = np.float64)
p, p_star, p_prime = np.zeros((3,nrows,mcols), dtype = np.float64)

# Face velocities
u_e, de_avg = np.zeros((2,nrows  ,mcols-1), dtype = np.float64)
v_n, dn_avg = np.zeros((2,nrows-1,mcols  ), dtype = np.float64)

# Diffusion Coeffs
D_x = (1/Re)*(dy/dx)
D_y = (1/Re)*(dx/dy)

b_u = (8.0/3.0)*D_y*U_lid*np.ones(mcols)

# Under-Relaxation Parameters
alpha_p   = 0.001
alpha_mom = 0.7
alpha_SOR = 1.0

# Iteration counts
max_iter_mom    = 20
max_iter_p      = 30
max_iter_simple = 5_000_000

# Convergence_criteria variables
b_mass_residual = 1.0
eps_mass_res = 1e-8
eps_mom_res = 1e-7

#%% Momentum Link Coeff generator
def MOMENTUM_LINK_COEFF_GEN(u, v, p_star, u_e, v_n):
    a_E, a_W, a_N, a_S, a_P = np.zeros((5,nrows,mcols), dtype = np.float64)
    S_avg_u, S_avg_v = np.zeros((2,nrows,mcols), dtype = np.float64)
    
    # For Interior cells
    a_E[1:-1,1:-1] = D_x - np.minimum(0, u_e[1:-1,1:  ])
    a_W[1:-1,1:-1] = D_x + np.maximum(0, u_e[1:-1, :-1])
    a_N[1:-1,1:-1] = D_y - np.minimum(0, v_n[1:  ,1:-1])
    a_S[1:-1,1:-1] = D_y + np.maximum(0, v_n[ :-1,1:-1])
    a_P[1:-1,1:-1] = 2*D_x + 2*D_y \
                   + np.maximum(0, u_e[1:-1,1:  ]) \
                   - np.minimum(0, u_e[1:-1, :-1]) \
                   + np.maximum(0, v_n[1:  ,1:-1]) \
                   - np.minimum(0, v_n[ :-1,1:-1])              
    
    S_avg_u[1:-1,1:-1] = 0.5 * dy *(p_star[1:-1,  :-2] - p_star[1:-1, 2:  ])
    S_avg_v[1:-1,1:-1] = 0.5 * dx *(p_star[ :-2, 1:-1] - p_star[2:  , 1:-1])


    # For Left Wall cells
    a_E[1:-1,0] = D_x - np.minimum(0, u_e[1:-1,0]) + D_x/3.0
    a_W[1:-1,0] = 0.0
    a_N[1:-1,0] = D_y - np.minimum(0, v_n[1:  ,0])
    a_S[1:-1,0] = D_y + np.maximum(0, v_n[ :-1,0])
    a_P[1:-1,0] = D_x + 2*D_y + 3.0*D_x \
                + np.maximum(0, u_e[1:-1,0])\
                + np.maximum(0, v_n[1:  ,0])\
                - np.minimum(0, v_n[ :-1,0])

    S_avg_u[1:-1,0] = 0.5 * dy *(p_star[1:-1,0] - p_star[1:-1,1])
    S_avg_v[1:-1,0] = 0.5 * dx *(p_star[ :-2,0] - p_star[2:  ,0])


    # For Right Wall cells
    a_E[1:-1,-1] = 0.0
    a_W[1:-1,-1] = D_x + np.maximum(0, u_e[1:-1,-1]) + D_x/3.0
    a_N[1:-1,-1] = D_y - np.minimum(0, v_n[1:  ,-1])
    a_S[1:-1,-1] = D_y + np.maximum(0, v_n[ :-1,-1])
    a_P[1:-1,-1] = D_x + 2*D_y + 3.0*D_x\
                 - np.minimum(0, u_e[1:-1,-1])\
                 + np.maximum(0, v_n[1:  ,-1])\
                 - np.minimum(0, v_n[ :-1,-1])
                 
    S_avg_u[1:-1,-1] = 0.5 * dy *(p_star[1:-1,-2] - p_star[1:-1,-1])
    S_avg_v[1:-1,-1] = 0.5 * dx *(p_star[ :-2,-1] - p_star[2:  ,-1])


    # For Top Wall cells
    a_E[-1,1:-1] = D_x - np.minimum(0, u_e[-1, 1:  ])
    a_W[-1,1:-1] = D_x + np.maximum(0, u_e[-1,  :-1])
    a_N[-1,1:-1] = 0.0
    a_S[-1,1:-1] = D_y + np.maximum(0, v_n[-1, 1:-1]) + D_y/3.0
    a_P[-1,1:-1] = 2*D_x + D_y + 3.0*D_y\
                 + np.maximum(0, u_e[-1,1:  ])\
                 - np.minimum(0, u_e[-1, :-1])\
                 - np.minimum(0, v_n[-1,1:-1])
    
    S_avg_u[-1,1:-1] = 0.5 * dy *(p_star[-1, :-2] - p_star[-1,2:  ])
    S_avg_v[-1,1:-1] = 0.5 * dx *(p_star[-2,1:-1] - p_star[-1,1:-1])


    # For Bottom Wall cells
    a_E[0,1:-1] = D_x - np.minimum(0, u_e[0,1:  ])
    a_W[0,1:-1] = D_x + np.maximum(0, u_e[0, :-1])
    a_N[0,1:-1] = D_y - np.minimum(0, v_n[0,1:-1]) + D_y/3.0
    a_S[0,1:-1] = 0.0
    a_P[0,1:-1] = 2*D_x + D_y + 3.0*D_y\
                + np.maximum(0, u_e[0,1:  ])\
                - np.minimum(0, u_e[0, :-1])\
                + np.maximum(0, v_n[0,1:-1])

    S_avg_u[0,1:-1] = 0.5 * dy *(p_star[0,  :-2] - p_star[0, 2:  ])
    S_avg_v[0,1:-1] = 0.5 * dx *(p_star[0, 1:-1] - p_star[1, 1:-1])


    # For Top-Left Corner cells
    a_E[-1,0] = D_x - min(0, u_e[-1,0]) + D_x/3.0
    a_W[-1,0] = 0.0
    a_N[-1,0] = 0.0
    a_S[-1,0] = D_y + max(0, v_n[-1,0]) + D_y/3.0
    a_P[-1,0] = D_x + D_y + 3.0*D_x + 3.0*D_y\
              + max(0, u_e[-1,0])\
              - min(0, v_n[-1,0])

    S_avg_u[-1,0] = 0.5 * dy *(p_star[-1,0] - p_star[-1,1])
    S_avg_v[-1,0] = 0.5 * dx *(p_star[-2,0] - p_star[-1,0])


    # For Bottom-Left Corner cells
    a_E[0,0] = D_x - min(0, u_e[0,0]) + D_x/3.0
    a_W[0,0] = 0.0
    a_N[0,0] = D_y - min(0, v_n[0,0]) + D_y/3.0
    a_S[0,0] = 0.0
    a_P[0,0] = D_x + D_y + 3.0*D_x + 3.0*D_y\
             + max(0, u_e[0,0])\
             + max(0, v_n[0,0])
    
    S_avg_u[0,0] = 0.5 * dy *(p_star[0,0] - p_star[0,1])
    S_avg_v[0,0] = 0.5 * dx *(p_star[0,0] - p_star[1,0])
    
    
    # For Top-Right Corner cells
    a_E[-1,-1] = 0.0
    a_W[-1,-1] = D_x + max(0, u_e[-1,-1]) + D_x/3.0
    a_N[-1,-1] = 0.0
    a_S[-1,-1] = D_y + max(0, v_n[-1,-1]) + D_y/3.0
    a_P[-1,-1] = D_x + D_y + 3.0*D_x + 3.0*D_y\
               - min(0, u_e[-1,-1])\
               - min(0, v_n[-1,-1])

    S_avg_u[-1,-1] = 0.5 * dy *(p_star[-1,-2] - p_star[-1,-1])
    S_avg_v[-1,-1] = 0.5 * dx *(p_star[-2,-1] - p_star[-1,-1])
    
    
    # For Bottom-Right Wall cells
    a_E[0,-1] = 0.0
    a_W[0,-1] = D_x + max(0, u_e[0,-1]) + D_x/3.0
    a_N[0,-1] = D_y - min(0, v_n[0,-1]) + D_y/3.0
    a_S[0,-1] = 0.0
    a_P[0,-1] = D_x + D_y + 3.0*D_x + 3.0*D_y\
              - min(0, u_e[0,-1])\
              + max(0, v_n[0,-1])

    S_avg_u[0,-1] = 0.5 * dy *(p_star[0,-2] - p_star[0,-1])
    S_avg_v[0,-1] = 0.5 * dx *(p_star[0,-1] - p_star[1,-1])

    d = alpha_mom/a_P

    if bicgstab_flag:
        S_avg_u[-1,:] = S_avg_u[-1,:] + b_u
        S_avg_u = alpha_mom*S_avg_u + (1-alpha_mom)*a_P*u
        S_avg_v = alpha_mom*S_avg_v + (1-alpha_mom)*a_P*v
    
        a_E = alpha_mom*a_E
        a_W = alpha_mom*a_W
        a_N = alpha_mom*a_N
        a_S = alpha_mom*a_S
    
        A = spdiags(data = np.array([a_P.reshape(nrows*mcols),
        -a_E.reshape(nrows*mcols),
        -a_W.reshape(nrows*mcols),
        -a_N.reshape(nrows*mcols),
        -a_S.reshape(nrows*mcols)]),
        diags = [0, -1, 1, -mcols, mcols])
        
        u_star = bicgstab(A = A,
                          b = S_avg_u.reshape(nrows*mcols),
                          x0 = u.reshape(nrows*mcols),
                          tol = eps_mom_res, 
                          maxiter=max_iter_mom)[0].reshape((nrows,mcols))
        
        v_star = bicgstab(A = A,
                          b = S_avg_v.reshape(nrows*mcols),
                          x0 = v.reshape(nrows*mcols),
                          tol = eps_mom_res, 
                          maxiter=max_iter_mom)[0].reshape((nrows,mcols))
    
    
        # Generating Hat velocities
        E = a_E * np.roll(u_star,-1,1)
        W = a_W * np.roll(u_star, 1,1)
        N = a_N * np.roll(u_star,-1,0)
        S = a_S * np.roll(u_star, 1,0)
    
        uP_hat = d*(E+W+N+S)
        uP_hat[-1,:] = uP_hat[-1,:] + d[-1,:]*b_u
    
        E = a_E * np.roll(v_star,-1,1)
        W = a_W * np.roll(v_star, 1,1)
        N = a_N * np.roll(v_star,-1,0)
        S = a_S * np.roll(v_star, 1,0)
    
        vP_hat = d*(E+W+N+S)
        
    else:
        u_star, uP_hat = GAUSS_SIEDEL_MOMENTUM(a_P, a_E, a_W, a_N, a_S, b_u, S_avg_u, u)
        v_star, vP_hat = GAUSS_SIEDEL_MOMENTUM(a_P, a_E, a_W, a_N, a_S, np.zeros(nrows), S_avg_v, v)
        

    return u_star, v_star, uP_hat, vP_hat, d

#%% Rhie Chow Momentum Interpolation
def RHIE_CHOW(u_e, v_n, uP_hat, vP_hat, d):
    de_avg[:,:] = 0.5*dy*(d[:,:-1] + d[:,1:])

    u_e[:,:] = 0.5*(uP_hat[:,:-1] + uP_hat[:,1:]) \
             + de_avg[:,:]*(p_star[:,:-1] - p_star[:,1:])

    dn_avg[:,:] = 0.5*dx*(d[:-1,:] + d[1:,:])

    v_n[:,:]  = 0.5*(vP_hat[:-1,:] + vP_hat[1:,:]) \
             + dn_avg[:,:]*(p_star[:-1,:] - p_star[1:,:])

    return u_e, de_avg, v_n, dn_avg

#%% Pressure Poisson Equation Solver
def PRESSURE_POISSON_EQUATION(p_prime, de_avg, dn_avg, u_e, v_n):
    ap_E, ap_W, ap_N, ap_S, ap_P = np.zeros((5,nrows,mcols), dtype = np.float64)
    b_mass_imbalance = np.zeros((nrows,mcols), dtype = np.float64)
    
    # For interior cells
    ap_E[1:-1,1:-1] = dy*de_avg[1:-1, 1:  ]
    ap_W[1:-1,1:-1] = dy*de_avg[1:-1,  :-1]
    ap_N[1:-1,1:-1] = dx*dn_avg[1:  , 1:-1]
    ap_S[1:-1,1:-1] = dx*dn_avg[ :-1, 1:-1]
    ap_P[1:-1,1:-1] = dy*de_avg[1:-1, 1:  ] + dy*de_avg[1:-1,  :-1]\
                    + dx*dn_avg[1:  , 1:-1] + dx*dn_avg[ :-1, 1:-1]
    
    b_mass_imbalance[1:-1,1:-1] = (-u_e[1:-1, 1:  ] + u_e[1:-1,  :-1])*dy\
                                + (-v_n[1:  , 1:-1] + v_n[ :-1, 1:-1])*dx

    # For Left Boundary cells
    ap_E[1:-1,0] = dy*de_avg[1:-1,0]
    ap_W[1:-1,0] = 0.0
    ap_N[1:-1,0] = dx*dn_avg[1:  ,0]
    ap_S[1:-1,0] = dx*dn_avg[ :-1,0]
    ap_P[1:-1,0] = dy*de_avg[1:-1,0] + dx*dn_avg[1:  ,0] + dx*dn_avg[ :-1,0]
    
    b_mass_imbalance[1:-1,0] = -u_e[1:-1,0] * dy\
                             +(-v_n[1:  ,0] + v_n[ :-1,0])*dx
    
    # For Bottom Boundary cells
    ap_E[0,1:-1] = dy*de_avg[0, 1:  ]
    ap_W[0,1:-1] = dy*de_avg[0,  :-1]
    ap_N[0,1:-1] = dx*dn_avg[0, 1:-1]
    ap_S[0,1:-1] = 0.0
    ap_P[0,1:-1] = dy*de_avg[0, 1:  ] + dy*de_avg[0, :-1] + dx*dn_avg[0, 1:-1]
    
    b_mass_imbalance[0,1:-1] = (-u_e[0,1:] + u_e[0,:-1])*dy\
                             - v_n[0,1:-1]*dx
    
    # For Right Boundary cells
    ap_E[1:-1,-1] = 0.0
    ap_W[1:-1,-1] = dy*de_avg[1:-1,-1]
    ap_N[1:-1,-1] = dx*dn_avg[1:  ,-1]
    ap_S[1:-1,-1] = dx*dn_avg[ :-1,-1]
    ap_P[1:-1,-1] = dy*de_avg[1:-1,-1] \
                  + dx*dn_avg[1:  ,-1] + dx*dn_avg[ :-1,-1]
    
    b_mass_imbalance[1:-1,-1] =   u_e[1:-1,-1]*dy \
                              + (-v_n[1:  ,-1] + v_n[:-1,-1])*dx
    
    # For Top Boundary cells
    ap_E[-1,1:-1] = dy*de_avg[-1,1:  ]
    ap_W[-1,1:-1] = dy*de_avg[-1, :-1]
    ap_N[-1,1:-1] = 0.0
    ap_S[-1,1:-1] = dx*dn_avg[-1,1:-1]
    ap_P[-1,1:-1] = dy*de_avg[-1,1:  ] + dy*de_avg[-1, :-1]\
                  + dx*dn_avg[-1,1:-1]
    
    b_mass_imbalance[-1,1:-1] = (-u_e[-1,1:] + u_e[-1,:-1])*dy\
                              + v_n[-1,1:-1]*dx
    
    # For Top-Left Corner cell
    ap_E[-1,0] = dy*de_avg[-1,0]
    ap_W[-1,0] = 0.0
    ap_N[-1,0] = 0.0
    ap_S[-1,0] = dx*dn_avg[-1,0]
    ap_P[-1,0] = dy*de_avg[-1,0] + dx*dn_avg[-1,0]
    
    b_mass_imbalance[-1,0] = -u_e[-1,0]*dy + v_n[-1,0]*dx
    
    # For Bottom-Left Corner cell
    ap_E[0,0] = dy*de_avg[0,0]
    ap_W[0,0] = 0.0
    ap_N[0,0] = dx*dn_avg[0,0]
    ap_S[0,0] = 0.0
    ap_P[0,0] = dy*de_avg[0,0] + dx*dn_avg[0,0]
    
    b_mass_imbalance[0,0] = -u_e[0,0]*dy - v_n[0,0]*dx
    
    # For Top-Right Corner cell
    ap_E[-1,-1] = 0.0
    ap_W[-1,-1] = dy*de_avg[-1,-1]
    ap_N[-1,-1] = 0.0
    ap_S[-1,-1] = dx*dn_avg[-1,-1]
    ap_P[-1,-1] = dy*de_avg[-1,-1] + dx*dn_avg[-1,-1]
    
    b_mass_imbalance[-1,-1] = u_e[-1,-1]*dy + v_n[-1,-1]*dx
    
    # For Bottom-Right Corner cell
    ap_E[0,-1] = 0.0
    ap_W[0,-1] = dy*de_avg[0,-1]
    ap_N[0,-1] = dx*dn_avg[0,-1]
    ap_S[0,-1] = 0.0
    ap_P[0,-1] = dy*de_avg[0,-1] + dx*dn_avg[0,-1]
    
    b_mass_imbalance[0,-1] = u_e[0,-1]*dy - v_n[0,-1]*dx
    
    if bicgstab_flag:
        A = spdiags(data = np.array([ap_P.reshape(nrows*mcols),
        -ap_E.reshape(nrows*mcols),
        -ap_W.reshape(nrows*mcols),
        -ap_N.reshape(nrows*mcols),
        -ap_S.reshape(nrows*mcols)]),
        diags = [0, -1, 1, -mcols, mcols])
        
        p_prime = bicgstab(A = A,
                          b = b_mass_imbalance.reshape(nrows*mcols),
                          x0 = p_prime.reshape(nrows*mcols),
                          tol = eps_mass_res,
                          maxiter=max_iter_p)[0].reshape((nrows,mcols))
    
    else:
        p_prime = GAUSS_SIEDEL_PPE(ap_P, ap_E, ap_W, ap_N, ap_S, b_mass_imbalance, p_prime)
    
    return p_prime, np.linalg.norm(b_mass_imbalance)

#%% Correctors
def VELOCITY_CORRECTORS(u_e, de_avg, v_n, dn_avg,
                        u, u_star, v, v_star,
                        p_prime, d):
    u_e[:,:] = u_e[:,:] + de_avg[:,:] * (p_prime[:,:-1] - p_prime[:,1:])

    v_n[:,:] = v_n[:,:] + dn_avg[:,:] * (p_prime[:-1,:] - p_prime[1:,:])
    
    # Cell Center Velocity Correction
    # Interior u velocity correction
    u[:,1:-1] = u_star[:,1:-1] \
              + 0.5 * d[:,1:-1] * dy * (p_prime[:,:-2] - p_prime[:,2:] )
    
    # Left wall u velocity correction
    u[:,0] = u_star[:,0] \
           + 0.5 * d[:,0] * dy * (p_prime[:,0] - p_prime[:,1])
           
    # Right wall v velocity correction
    u[:,-1] = u_star[:,-1] \
            + 0.5 * d[:,-1] * dy * (p_prime[:,-2] - p_prime[:,-1])

    # Interior v velocity correction
    v[1:-1,:] = v_star[1:-1,:] \
              + 0.5 *d[1:-1,:] * dx * (p_prime[:-2,:] - p_prime[2:,:])
      
    # Bottom Wall v velocity correction
    v[0,:] = v_star[0,:] \
           + 0.5 *d[0,:] * dx * (p_prime[0,:] - p_prime[1,:])
           
    # Top Wall v velocity correction
    v[-1,:] = v_star[-1,:] \
            + 0.5 *d[-1,:] * dx * (p_prime[-2,:] - p_prime[-1,:])

    return u_e, v_n, u, v

#%% Gauss Siedel
@njit
def GAUSS_SIEDEL_MOMENTUM(a_P, a_E, a_W, a_N, a_S, b, S_avg, phi):
    phi_star = np.copy(phi)
    phi_hat = np.copy(phi)
    for i in range(max_iter_mom):
        for row in range(nrows):
            for col in range(mcols):
                
                if col == mcols-1:
                    E = 0
                else:
                    E = a_E[row,col] * phi_star[row,col+1]
                
                if col == 0:
                    W = 0
                else:
                    W = a_W[row,col] * phi_star[row,col-1]
                
                if row == nrows-1:
                    N = 0
                else:
                    N = a_N[row,col] * phi_star[row+1,col]
                
                if row == 0:
                    S = 0
                else:
                    S = a_S[row,col] * phi_star[row-1,col]
                    
                if row == nrows-1:
                    B = b[col]
                else:
                    B = 0.0

                phi_hat[row,col] = alpha_mom*(E + W + N + S + B)/a_P[row,col]
                phi_star[row,col] = phi_hat[row,col] \
                                  + alpha_mom*S_avg[row,col]/a_P[row,col] \
                                  + (1-alpha_mom)*phi[row,col]
    return phi_star, phi_hat

@njit
def GAUSS_SIEDEL_PPE(a_P, a_E, a_W, a_N, a_S, RHS, phi):
    phi_star = np.copy(phi)
    
    for i in range(max_iter_mom):
        for row in range(nrows):
            for col in range(mcols):
                
                if col == mcols-1:
                    E = 0
                else:
                    E = a_E[row,col] * phi_star[row,col+1]
                
                if col == 0:
                    W = 0
                else:
                    W = a_W[row,col] * phi_star[row,col-1]
                
                if row == nrows-1:
                    N = 0
                else:
                    N = a_N[row,col] * phi_star[row+1,col]
                
                if row == 0:
                    S = 0
                else:
                    S = a_S[row,col] * phi_star[row-1,col]

                
                phi_star[row,col] = alpha_SOR*(E + W + N + S + RHS[row,col])/a_P[row,col]

    return phi_star

#%% SIMPLE ITERATIONS
iter_simple = 0
while (iter_simple < max_iter_simple and b_mass_residual > eps_mass_res):
    u_star, v_star, uP_hat, vP_hat, d = \
        MOMENTUM_LINK_COEFF_GEN(u, v, p_star, u_e, v_n)
    
    u_e, de_avg, v_n, dn_avg = \
        RHIE_CHOW(u_e, v_n, uP_hat, vP_hat, d)
    
    p_prime, b_mass_residual = \
        PRESSURE_POISSON_EQUATION(p_prime, de_avg, dn_avg, u_e, v_n)
        
    u_e, v_n, u, v = \
        VELOCITY_CORRECTORS(u_e, de_avg, v_n, dn_avg,
                            u, u_star, v, v_star,
                            p_prime, d)
    
    p = p_star + alpha_p*p_prime
    
    u_star = np.copy(u)
    v_star = np.copy(v)
    p_star = np.copy(p)
    
    iter_simple += 1
    if iter_simple % 1000 == 0:
        print(b_mass_residual)

#%% Post process

import plotly.figure_factory as ff
import plotly.io as pio
pio.renderers.default = 'browser'
ff.create_streamline(x_coll, y_coll, u, v, arrow_scale=0.01).show()