import time
import numpy as np
import cmath
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.linalg import block_diag
from dipolar_Bogoliubov import *

a = 1.0
Vext = -10
k_mod = np.pi
eps = 0.05
g = +0.5
Vext, q = 8.2, np.pi/2
#Vext =0
q_index = int(q*L/(2*np.pi)) +L//2
#No the way I do it I couple Q -k,k with Q-(k\pm q), k\pm q
#Normally if g=0 there is no coupling between impurity and bosons and should be bare impurity spectrum


Q_vals = 2*np.pi*(np.arange(L)-L//2)/L 


def k_to_index(k, L):
    dk = 2*np.pi / L
    return int(np.round(k / dk + L//2)) % L

def spec_impurity(k):
    return -2*np.cos(k*a) +2 

def bogo_spec(k):
    return omega_k[k_to_index(k, L)]

def boson_groundstate(V_ext):
    k_vals = 2*np.pi*np.arange(L)/L
    #epsilon_k = -2*t*np.cos(k_vals)  # dispersion hopping
    epsilon_k =[bogo_spec(k) for k in k_vals ]

    H = np.diag(epsilon_k)
    for k in range(L):
        H[k, (k + q_index) % L] += V_ext/2
        H[k, (k - q_index) % L] += V_ext/2

    eigvals, eigvecs = np.linalg.eigh(H)
    
    phi0 = eigvecs[:, 0]
    
    return phi0

    
def uk(k):
    idx = k_to_index(k,L)
    Ek_loc = Ek[idx]
    #return cmath.sqrt((Ek_loc + n0*V0 + omega_k[idx]) / (2*omega_k[idx]))
    #return cmath.sqrt((Ek_loc + 2*n0*V0 + omega_k[idx]) / (2*omega_k[idx]))
    return np.sqrt((Ek_loc + 2*n0*V0 + omega_k[idx]) / (2*omega_k[idx]))


def vk(k): # BECOMES IMAGINARY SOMETIMES NO chemical pot problem
    idx = k_to_index(k, L)
    Ek_loc = Ek[idx]
    #print("Ek_loc is", Ek_loc)
    #print("(Ek_loc + n0*V0 - omega_k[idx]) is", (Ek_loc + n0*V0 - omega_k[idx]))
    #print('omega is ', (2*omega_k[idx]))
    #return cmath.sqrt((Ek_loc + n0*V0 - omega_k[idx]) / (2*omega_k[idx]))
    #return cmath.sqrt((Ek_loc + 2*n0*V0 - omega_k[idx]) / (2*omega_k[idx]))
    return np.sqrt((Ek_loc + 2*n0*V0 - omega_k[idx]) / (2*omega_k[idx]))

def vk_pow2(k):
    return uk(k)**2 -1
cste = g/L*sum( vk_pow2(2*np.pi/L * k1) for k1 in range(1,L))
Omega_vals = np.linspace(cste-50 ,cste + 20 , 800)

Omega_vals = np.linspace(-10,15 , 800)


def Wk(k):
    return uk(k) + vk(k)

def V1kkprime(k, k_prime):
    return uk(k)*uk(k_prime) + vk(k)*vk(k_prime)

def Frolich_Hamiltonian(p, g, L):
    """
    p : total polaron momentum
    g : impurity boson coupling
    L : number of lattice sites
    """
    H = np.zeros((L, L), dtype=np.complex128)
    #cste = g/L * sum( vk(2*np.pi/L * k1)**2 for k1 in [i for i in range(L) if i!=L//2])
    cste = g/L * sum( vk_pow2(2*np.pi/L * k1) for k1 in range(1,L))
    #for k1 in range(L):
    #    print('vk **2 is ', vk_pow2(2*np.pi/L * k1))
    print('cste is ', cste)
    print('Here ')
    for k in range(L): # momentum from 0 to 2pi
        kk = 2*np.pi/L*k
        for kp in range(L):
            kkp = 2*np.pi/L*kp
            # impurity kinetic energy
            impurity_kin =  spec_impurity(p - kk) if k == kp else 0
            # phonon energy
            bath =  bogo_spec(kk) if (k == kp and k != 0) else  0
            # mean-field shift
            BI_int_cste1 = g*n0 if k == kp else 0
            BI_int_cste2 = cste if k == kp else 0
            # linear Fröhlich term
            BI_int_W = 0
            if k == 0 and kp != 0:
                #BI_int_W = g * cmath.sqrt(n0 / L) * Wk(kkp)
                BI_int_W = g * np.sqrt(n0 / L) * Wk(kkp)
            elif k != 0 and kp == 0:
                #BI_int_W = g * cmath.sqrt(n0 / L) * Wk(kk)
                BI_int_W = g * np.sqrt(n0 / L) * Wk(kk)
            #print('BI W is', BI_int_W )
            # two-phonon vertex
            BI_int_V =  g/L*V1kkprime(kk, kkp) if (k != 0 and kp != 0) else 0
            #print('BI W is', BI_int_W )
            
            
            H[k, kp] =  impurity_kin + bath + BI_int_cste1 + BI_int_cste2 + BI_int_W + BI_int_V
            # external cosine potential:
            #if np.isclose((kkp - kk) % (2*np.pi), k_mod) or np.isclose((kkp - kk) % (2*np.pi), -k_mod):
            #    H[k, kp] += Vext/2

    return H

def Frolich_Hamiltonian_broken_translation_full(g, L, Vext, q_index):
    """
    Full Frohlich Hamiltonian in the momentum-space basis without translation symmetry.
    The basis is |Q_imp, k_phonon>, total size L^2 x L^2.
    g       : impurity-boson coupling
    L       : number of momentum states for impurity and phonon
    Vext    : amplitude of external potential (sine/cosine)
    q_index : momentum index of external potential (integer)
    """
    blocks = [ Frolich_Hamiltonian(p, g, L) for p in Q_vals] #Note that Q vals goes from -pi to pi
    H= block_diag(*blocks)
    print('size of total H', H.shape)
    print('Len Q_vals',len(Q_vals))
    print('Len list block',len(blocks))
    print('shape block',blocks[0].shape)

    #H = np.zeros((L*L, L*L), dtype=np.complex128)
    # convert 2D indices to 1D
    def idx(Q_imp, k_phonon):
        return Q_imp*L + k_phonon
    num_Q = len(Q_vals)
    # Basis Q is total momentum and k is phonon momentum so it makes sense that different Q but Q-k should remain the same...
    for Q1_idx, Q1 in enumerate(Q_vals): #Q is the total momentum in the basis I chose and correspond to momentum Q-k for impurity and k for phonons
        for delta in [q_index, -q_index +L]:  # instead of looking for all k_primes... cos(qx) couples ±q this plus and minus index is odd because it is 
            
            for k_1 in range(L):
                k_2 =  (k_1 + delta)%L
                Q2_idx = (Q1_idx  -k_1 +  k_2)%L
                #Q2_idx = int((Q1_idx + delta) % num_Q) # between 0 and L-1
                #Q2_idx2 = Q1_idx 
                ind1 = idx(Q1_idx, k_1)
                ind2 = idx(Q2_idx, k_2 )
                H[ind1, ind2] += Vext/2
    return H

def spectral_fct_full(Omega_vals, g, L, Vext, q_index, eps):
    t_start = time.time()
    H =  Frolich_Hamiltonian_broken_translation_full(g, L, Vext, q_index)
    print('before diag')
    eigvals, eigvecs = np.linalg.eigh(H)  # eigenvectos[:,i]
    print('after diag')
    #Now the first index labels the Q sector I am looking at like before and the second the momenta of the phonon
    #and the last index is the number of eigenvectors
    eigvecs =  eigvecs.reshape((L,L,L**2))
    # ATTENTION je dois choisir le bon etat fondamental pour mon bain de bosons
    #oscillator_strengths = np.abs(eigvecs[:, 0, :])**2  # (Q, n) #first component is the bare impurity part because Frolich Hamiltonian goes from 0 to 2pi
    # boson GS
    phi0 = boson_groundstate(Vext)  # Boson GS in k basis from  0 to 2pi
    print('phi groun state is', phi0)
    #oscillator_strengths = np.abs(np.tensordot(phi0.conj(), eigvecs[:, :, :], axes=([0],[1])))**2
    oscillator_strengths = np.zeros((L, L**2), dtype=np.float64)  # (Q, n)

    for Q_idx, Q in enumerate(Q_vals): # Polaron momentum from -pi to pi ok bc in the big matrix and in the wavefunction we oganise by Q then by k where Q goes from -pi to pi and k from 0 to 2pi
        for n in range(L**2):  #eigenvector number
            overlap = 0.0
            for q in range(L):  #phonon momentum
                k_idx = (Q_idx + q +L//2) % L -L//2
                overlap += phi0[q].conj()*eigvecs[k_idx, q, n]
            oscillator_strengths[Q_idx, n] = np.abs(overlap)**2
    #oscillator_strengths = np.abs(eigvecs[:, 0, :])**2 # Works only when V_ext =0
    Omega_vals = Omega_vals[:, None, None]     # (Ω, 1, 1)
    print('shape omega vals',Omega_vals.shape )
    #eigvals = eigvals[None, :, :]  # (1, Q, n)
    eigvals = eigvals[None, None, :]  # (1, Q, n)
    print('shape eigvals',eigvals.shape )
    osc = oscillator_strengths[None, :, :]     # (1, Q, n)
    print('shape osc',osc.shape )
    denom = Omega_vals - eigvals + 1j*eps
    print('shape denom',denom.shape )
    spectral_contributions = osc / denom  #(Ω, Q, n)
    spectral_vals = -2*np.sum(spectral_contributions.imag, axis=2)#(Ω, Q) OK
    print(f"Spectral function computed in {time.time() - t_start:.2f} s")
    return spectral_vals

def spectral_function_table(Q_vals, Omega_vals, g, L, eps, Frolich_Hamiltonian):
    t_start = time.time()
    num_Q = len(Q_vals)
    num_Omega = len(Omega_vals)
    eigvals_list = []
    eigvecs_list = []
    for Q in Q_vals:
        print("Polaron Q =", Q)
        H = Frolich_Hamiltonian(Q, g, L)
        print('is H hermitian ?', np.max(np.abs(H - H.T.conj())))
        print("Diagonalizing...")
        eigvals, eigvecs = np.linalg.eigh(H)
        eigvals_list.append(eigvals)
        eigvecs_list.append(eigvecs)
        print('eigenvals',eigvals)
    eigvals_array = np.array(eigvals_list)              # (Q, n)
 #  print('eigvals array',eigvals_array)
    eigvecs_array = np.stack(eigvecs_list, axis=0)      # (Q, n, n)
 #  print('eigvals array',eigvals_array)
    # overlap with bare impurity (k=0 state)
    oscillator_strengths = np.abs(eigvecs_array[:, 0, :])**2  # (Q, n) #first component is the bare impurity part because Frolich Hamiltonian goes from 0 to 2pi
 #  print('oscillaor strength',oscillator_strengths)
    Omega_vals = Omega_vals[:, None, None]     # (Ω, 1, 1)
    eigvals_array = eigvals_array[None, :, :]  # (1, Q, n)
 #  print('eigvals array',eigvals_array)
    osc = oscillator_strengths[None, :, :]     # (1, Q, n)
    denom = Omega_vals - eigvals_array + 1j*eps
  # print('denom is ',denom)
  # print('osc is ',osc)
    spectral_contributions = osc / denom

    spectral_vals = -2*np.sum(spectral_contributions.imag, axis=2)

    print(f"Spectral function computed in {time.time() - t_start:.2f} s")

    return spectral_vals
def spectral_function_plot_table(Q_vals, Omega_vals, spectral_vals, g, L, eps):
    Q_mesh, Omega_mesh = np.meshgrid(Q_vals, Omega_vals, indexing='ij')

    vmin = max(spectral_vals.min(), 1e-6)
    vmax = spectral_vals.max()

    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

    plt.figure(figsize=(8, 6))
    plt.pcolormesh(
        Q_mesh,
        Omega_mesh,
        spectral_vals.T,
        cmap='viridis',
        norm=norm,
        shading='auto'
    )

    cbar = plt.colorbar()
    cbar.set_label(r"$\mathcal{A}(\Omega,Q)$", fontsize=14)

    plt.xlabel(r"Impurity momentum $Q$", fontsize=15)
    plt.ylabel(r"$\Omega$", fontsize=15)
    plt.title(
        rf"$g={g}$, $L={L}$, $V={V}$, $Vext={Vext}$, $qext={round(q/np.pi,2)}\pi$, $n_0={round(n0,3)}$, $\Gamma={eps}$, $\alpha=${alpha}, $\delta$={delta}",
        fontsize=14
    )

    plt.tight_layout()
    plt.show()

  

#A = spectral_function_table( Q_vals, Omega_vals, g, L, eps, Frolich_Hamiltonian)
A = spectral_fct_full(Omega_vals, g, L, Vext, q_index, eps)
print('max and min', np.max(A),np.min(A),'voila')
print('A', A)
spectral_function_plot_table(Q_vals, Omega_vals, A, g, L, eps)


