import numpy as np
import time
from quspin.basis import tensor_basis, boson_basis_1d, boson_basis_general # Hilbert space boson basis
from scipy.linalg import norm


def translate(state, r):
    """Translate a state array by r sites (cyclic)."""
    return state[-r:] + state[:-r]
def embed_basis_vector(v, k,basis_k, basis_full, L):
    """
    Embed a single vector from the basis_k representative into the full basis.
    v : integer that labels the basis state from basis_k
    """
    phi_b = np.zeros(basis_full.Ns, dtype=np.complex128)
    for r in range(L):
        v_string = basis_k.int_to_state(v)[1:-1]
        v_string = v_string.replace(" ", "")
        state_r = translate(v_string, r)
        #print('state r', state_r)
        idx = basis_full.index(state_r)
        phi_b[idx] += np.exp(-1j*k*2*np.pi/L*r) / np.sqrt(L)
    return phi_b / norm(phi_b)

def embed_basis(k, basis_k, basis_full, L):
    """Embed all basis_k representatives into full basis."""
    embedded_basis_states_k = [None] * basis_k.Ns
    for i, state in enumerate(basis_k.states): #state is an integer
        embedded_basis_states_k[i] = embed_basis_vector(state, k,basis_k, basis_full, L)
    return embedded_basis_states_k

def embed_vector(v, k, basis_k, basis_full, L):
    """Embed a vector in the k-basis into full basis.
    assume v is a list of number, a weigth for each basis element"""
    embedded_basis_states = embed_basis(k, basis_k, basis_full, L)
    embedded_state = np.zeros(basis_full.Ns, dtype=np.complex128)
    for coeff, state_vec in zip(v, embedded_basis_states):
        embedded_state += coeff * state_vec
    return embedded_state


#N,L=2,3

#sps = N+1
# Translational symmetry
#T = np.roll(np.arange(L), -1)
#allowed momenta
#n_range = np.arange(-(L//2), L//2 + (L % 2))
#qvals = 2*np.pi*n_range/L

#k=1

#basis_full = boson_basis_general(N=L, Nb=N, sps = sps) 
#basis_k = boson_basis_general(N=L, Nb=N, sps =sps, kxblock=(T, k))
#basis_1 = boson_basis_general(N=L, Nb=N, sps =sps, kxblock=(T, 1))
#basis_2 = boson_basis_general(N=L, Nb=N, sps =sps, kxblock=(T, 2))
#print('basis full',basis_full)
#print('basis k',basis_k)
#v = basis_k.states[0]
#print('v is', v)
#vec = embed_basis_vector(v, k,basis_k, basis_full, L)
#print('vec',vec)


#v = np.zeros(basis_k.Ns,dtype = complex)
#v[0] =1
#full_v = embed_vector(v, k, basis_k, basis_full, L)

#full_v2= basis_full.project_to(basis_k.project_from(v,sparse=False),sparse=False)
#full_v3= basis_k.project_from(v,sparse=False)
#print('size of full_v2',len(full_v2))
#print('size of full_v3',len(full_v3))
#print('DIY embedding \n', full_v)
#print('projector embedding \n', full_v2)

#print(embed_basis(k, basis_k, basis_full, L)[0])
#print('diff', full_v - full_v2)
#print('diff', np.max(np.abs(full_v - full_v2)))