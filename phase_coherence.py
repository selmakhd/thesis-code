from Lanczos import *
from find_polaron_wavefunction import *
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh


L = 16
N = 2
n = N/L
J=1
mu = -2
U = 0 #Contact
U2 = 0 #next neighbor interaction
Vc = 0 # ext pot amplitude for the power law potential
alpha = 3  #the power of the power law potential
sps = None

g=-5
Gamma = 0.05
M = 29 # Lanczos iterations
# Translational symmetry
T = np.roll(np.arange(L), -1)
#allowed momenta
n_range = np.arange(-(L//2), L//2 + (L % 2))
qvals = 2*np.pi*n_range/L



def one_BRDM(w_window,K,N,U,U2,Vc,alpha,g,mu,J,T,L = L, tolerance =0.2,sps = None,M=M):
    if sps == None:
        sps = N+1
    Psi_K,psiw,psik,k_int,full_basis,Z  = Psi_polaron(w_window,K,N,U,U2,Vc,alpha,g,mu,J,T,L = L,tolerance = tolerance, sps =sps,M=M)
    print('psi K', Psi_K)
    bath_dressing = full_basis.partial_trace(Psi_K, sub_sys_A="left",return_rdm='A', enforce_pure=True, sparse=False)
    bath_basis = full_basis.basis_left
    corr = np.zeros((L,L),dtype = complex)
    for j in range(L):
        for i in range(L):
            corrji = [['+-', [[1,i,j,]]]]
            static =corrji
            corr_ji_op= hamiltonian(static, [], basis=bath_basis, dtype=np.complex128, check_pcon=False, check_symm=False,check_herm=False)
            corr[j,i] = corr_ji_op.expt_value(bath_dressing)
        print('ok done')
    print('corr is', corr)
    return corr, psiw,k_int,Z

def plot_one_BRDM(corr,psiw,k_int,Z, num_levels=50, gamma=0.5, title='Three-body correlation'):
    """
    corr : 2D np.array, shape (L,L)  matrix
    num_levels : int Number of levels for colormap normalization (if needed)
    gamma : float PowerNorm gamma for color scaling
    title : str Plot title
    """
    L = corr.shape[0]

    i_vals, j_vals = np.meshgrid(np.arange(L), np.arange(L))
    i_vals = i_vals.flatten()
    j_vals = j_vals.flatten()
    corr_vals = np.real(corr).flatten() 

    sizes = 3000 * np.abs(corr_vals) / np.max(np.abs(corr_vals))

    norm = PowerNorm(gamma=gamma, vmin=np.min(corr_vals), vmax=np.max(corr_vals))

    plt.figure(figsize=(6,6))
    sc = plt.scatter( i_vals,j_vals, s=sizes, c=corr_vals, cmap='viridis',   norm=norm,    alpha=0.8,    edgecolor='k' )

    plt.colorbar(sc, label=r'|<a_i^\dagger a_j>|')
    plt.xlabel('Site i')
    plt.ylabel('Site j')
    plt.title(title + fr'E={round(psiw,3)}, k_int={k_int}, Z={round(Z,4)}, N={N}, L={L}, U={U}, U2={U2}, Vc={round(Vc,2)}, $\alpha=${alpha}, g={g}, sps={sps}' )
    plt.xticks(np.arange(L))
    plt.yticks(np.arange(L))
    plt.gca().set_aspect('equal')
    plt.show()

def momentum_distribution(phase_coh_matrix):
    Dk = np.zeros(L)
    i1, i2 = np.meshgrid(np.arange(L), np.arange(L), indexing='ij')
    for i in range(L):
        Dk[i]=1/L*np.sum( np.exp(1j*(i1 -i2)*qvals[i])*phase_coh_matrix[i1,i2])
    return Dk

w_window = (-3,-0.5)
w_window = (0,2.5)
K =np.pi
phase_coh,psiw,k_int,Z=one_BRDM(w_window,K,N,U,U2,Vc,alpha,g,mu,J,T,L = L, tolerance =0.2,sps = sps,M=M)

Distrib_k = momentum_distribution(phase_coh)
plt.bar(qvals, Distrib_k.real, width=qvals[1]-qvals[0])
plt.xlabel(r"$k$")
plt.ylabel(r"$n(k)$")
plt.show()


E,V = np.linalg.eigh(phase_coh)
print('highest eigenvalues', E[-1],' and ', E[-2])
print('eigenvalue associated to highest', V[:,-1])
plot_one_BRDM(np.angle(phase_coh),psiw,k_int,Z, num_levels=50, gamma=0.5, title='Phase coherence')


