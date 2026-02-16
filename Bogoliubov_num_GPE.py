from GPE_dipolar import *
import numpy as np
import math

L = 12            # number of sites
V = 1.6           # dipolar interaction strength
U = 0        # to enforce hardcore boson condition 
N = 4             # condensate density (number per site)
J = 1.0           # hopping amplitude
n0 = N/L

#momenta (PBC)
k_vals = 2*np.pi*(np.arange(L)-L//2)/L 

def chemical_potential(psis,J,V,U):
    nb_bosons = round((np.conj(psis)@psis).real,0)
    return (Kinetic(psis,J) + 2*Interac(psis,V,U)) / nb_bosons

def T_matrix_canonical_basis(pow,L):
    A = np.zeros((L,L),dtype=complex)
    for k in range(L):
        A[k,(k+pow)%L] = 1
    zero = np.zeros_like(A)
    return np.block([[A, zero],[zero, A.conj()]])

def A_Real(J,V,U,psi_GS):
    L = len(psi_GS)
    A = np.zeros((L,L),dtype=complex)
    mu = chemical_potential(psi_GS,J,V,U)
    n = np.abs(psi_GS)**2
    U_dip = np.zeros(L)
    for j in range(L):
        for l in range(L):
            if l != j:
                r = f(abs(j-l), L)
                U_dip[j] += n[l]/(r**3)

    for j in range(L):
        A[j,j] = -2*U*n[j] + mu
        A[j,j] += V*U_dip[j]/n0**3        
        A[j,(j+1)%L] = J
        A[j,(j-1)%L] = J
    return A

def B_Real(V,U,psi_GS):
    L = len(psi_GS)
    B = np.zeros((L,L),dtype=complex)
    for j in range(L):
        B[j,j] = -U*psi_GS[j]**2
    return B

def construct_L(A,B,L,pow=1,epsilon=1e-6):
    zero = np.zeros_like(A)
    Translation = T_matrix_canonical_basis(pow,L)
    Lmat = np.block([
        [-A, -B],
        [ B.conj(), A.conj()] ])
    return Lmat + 2*epsilon*(Translation - Translation.conj().T)


#psis_GS = np.ones(L)*np.sqrt(n0) 
psis_GS= GPE_condensate(L,N,V,U,J =1,fluc_max = 0, phi=0,t0=0,step = 0.001,n_steps = 5000)[-1]
psis_GS_R = psis_GS                  
pow = 1
A_R = A_Real(J,V,U,psis_GS_R)
B_R = B_Real(V,U,psis_GS_R)

#stationarity check
stationary = np.max(np.abs(A_R @ psis_GS_R + B_R @ np.conj(psis_GS_R)))
print("BdG stationarity:", stationary)

eom_matrix = construct_L(A_R,B_R,L,pow)

Translation = T_matrix_canonical_basis(pow,L)
print('commutator ladder 1D', np.max(np.abs(eom_matrix @ Translation - Translation @ eom_matrix)))

#DIAGONALIZE 
e_vals_unsorted, e_vecs_unsorted = np.linalg.eig(eom_matrix)
idx = np.argsort(e_vals_unsorted.real)
e_vals = e_vals_unsorted[idx].real
e_vecs = e_vecs_unsorted[:,idx]

def symplectic_normalization(uv_matrix):
    u = uv_matrix[:L,:]
    v = np.conj(uv_matrix[L:,:])
    norm = np.sqrt(np.sum(np.abs(u)**2,axis=0)  - np.sum(np.abs(v)**2,axis=0))
    return u/norm, v/norm

u_matrix, v_matrix = symplectic_normalization(e_vecs[:,1:])  # skip Goldstone
ug, vg = u_matrix[:,0], v_matrix[:,0]

print('SYMPLECTIC NORM FIRST MODE', np.sqrt(np.sum(np.abs(ug)**2) - np.sum(np.abs(vg)**2)))
