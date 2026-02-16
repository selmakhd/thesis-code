from scipy.linalg import eigh_tridiagonal, norm
from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import tensor_basis, boson_basis_1d, boson_basis_general # Hilbert space boson basis
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import time


L=22            #system size
N=3             #number of bosons
a=1             #lattice spacing
n = N/L
J=1.0           #hopping
U=2.5           #boson-boson interaction
mu=-2           #chem potential
g=-5.0          #impurity-boson interaction
Gamma=0.05      #Broadening
M = 67          #Lanczos iterations


#translational symmetry
T = np.roll(np.arange(L), -1)
#allowed momenta
n_range = np.arange(-L//2, L//2 + (L % 2))
qvals = 2*np.pi*n_range/L

def free_spec(k,mu):
    return -2*np.cos(k*a) -mu 
def doublon_energy(U, k,J):
    E_trial = np.linspace(4,11,500)
    LS = np.array([1/L*sum(1/(E -4+4*J*np.cos(k/2)*np.cos(qvals))) for E in E_trial ])
    best_index = np.argmin( np.abs(LS -1/U)  )
    return E_trial[best_index]

def bath_Hamil(L,N,U,mu,J,k= None,sps = N+1):
    """ Outputs the bath hamiltonian and the basis within the specified momentum sector k """
    basis = boson_basis_general(N=L, Nb=N, sps = sps) if k== None else  boson_basis_general(N=L, Nb=N, sps =sps, kxblock=(T, k))
    bath_hop=[[-J,i,(i+1)%L] for i in range(L)]
    bath_interact=[[0.5*U,i,i] for i in range(L)]
    bath_pot=[[-mu-0.5*U ,i] for i in range(L)]
    bath_static = [['+-', bath_hop], ['-+', bath_hop], ['n', bath_pot], ['nn', bath_interact]]
    H0 = hamiltonian(bath_static,[], basis=basis, dtype=np.complex128) 
    return H0,basis


def polaron_Hamil(L,N,U,g,mu,J,k= None,sps = N+1, q=None):
    """ Gives me the hamiltonian of the bath + impurity and the basis within the specified momentum sector k """
    bath_basis = boson_basis_general(N=L, Nb=N, sps = sps) if k== None else  boson_basis_general(N=L, Nb=N, sps =sps, kxblock=(T, k))
    imp_basis  = boson_basis_general(N=L, Nb=1,sps = 2 ) if q== None else  boson_basis_general(N=L, Nb=1, sps =2, kxblock=(T, q))
    tensored_basis = tensor_basis(bath_basis, imp_basis)

    hop=[[-J,i,(i+1)%L] for i in range(L)]
    bath_interact=[[0.5*U,i,i] for i in range(L)]
    bath_pot=[[-mu-0.5*U  ,i] for i in range(L)]
    imp_chem=[[-mu,i] for i in range(L)]

    bath_static = [['+-|', hop], ['-+|', hop], ['n|', bath_pot], ['nn|', bath_interact]]
    imp_static = [['|+-', hop], ['|-+', hop], ['|n', imp_chem]]
    imp_bath_int = [['n|n', [[g,j,j] for j in range(L)]]]

    static = bath_static + imp_static + imp_bath_int

    polaron_H = hamiltonian(static, [], basis=tensored_basis, dtype=np.complex128, check_pcon=False, check_symm=False)

    return tensored_basis, polaron_H

def lanczos_core(Psi0, H, V, M=100):
    """ Input 
        Psi0 : initial state in a Quspin basis (much larger than M),  
        H : Hamiltonian in Quspin basis, 
        V: a bunch of column vectors expressed in the Krylov basis,
        M : number of Lanczos iterations
        Output
        A = [a_0,...,a_{M}] the diagonal coefficients
        B = [0,b_1,...,b_{M}] the off-diagonal coefficients from wich you build H' within Krylov subspace

        Remark : I assume that they are all linearly independant though... Otherwise the a b coeff are ill-defined
      """
    VV = np.zeros((len(Psi0), V.shape[1]), dtype=Psi0.dtype) # Ns lines and columns equal to number of vectors i wanna convert 
    A = np.zeros(M +1, dtype=Psi0.dtype)
    B = np.zeros(M +1, dtype=Psi0.dtype)
    #I need to keep in memory 3 vectors at a time
    #Initialisation
    phi0 = Psi0.copy() #the only one i need to define a_0  and  b_0
    phi_minus_1 = np.zeros_like(Psi0)
    b1 = norm(phi0) 
    B[0] = 0
    A[0] = np.vdot(phi0,H.dot(phi0)) / np.vdot(phi0, phi0) 
    VV += np.outer(phi0/norm(phi0), V[0, :])
    for n in range(M):
        phi_plus_1 = H.dot(phi0) - A[n]*phi0 - B[n]**2*phi_minus_1
        normphi = norm(phi_plus_1)
        if normphi !=0 :
            B[n+1] = normphi / norm(phi0)
            A[n+1] = np.vdot(phi_plus_1, H.dot(phi_plus_1)) / normphi**2 
        else : 
            raise ValueError("the Krylov basis is not independant")
        phi_minus_1 = phi0.copy()
        phi0 = phi_plus_1.copy()
        VV += np.outer(phi0 / norm(phi0), V[n+1, :]) #because H is expressed in the normalized basis
    B[0] = norm(Psi0) #useful for the rationalized fraction, if the initial state is chosen the right way
    return A.real, B.real, VV

def lanczos_AB(Psi0, H, M=100):
    A, B, _ = lanczos_core(Psi0, H, np.zeros((M+1, 1)), M=M)
    return A, B

def continued_fraction(w, A, B):
    c = np.zeros_like(w, dtype=np.complex128)
    for n in reversed(range(len(A))):
        c = B[n]**2 / (w - A[n] - c)
    return c

def lanczos_green(w, Psi0, H, M=100, Gamma=0.05, E= 0):
    A, B = lanczos_AB(Psi0, H, M=M)
    A = A -E       #DSF shift
    return continued_fraction(w + 1j*Gamma, A, B)

def lanczos_gs(Psi0, H, M=100, nev=1):
    A, B = lanczos_AB(Psi0, H, M=M)
    E, V = eigh_tridiagonal(A, B[1:])
    _, _, VV = lanczos_core(Psi0, H, V[:, :nev], M=M)
    return E[:nev], VV

def full_procedure_per_qsector(H,basis,L,N,U,mu,J, Green_op,w,q,k =0,sps =N+1, M= 100, Gamma = 0.05):
    """  I have to run the Lanczos algorithm once  with some intial state in the right sector
        to get the ground state. Then I rerun it again with an intial state psi0 = hat{a}|GS>
        Input
        H,basis : hamiltonian and basis of the bosons in the momentum sector of the GS
        L,N,U,mu,J : system parameters
        k : momentum sector for the GS
        q : momentum carried by Green_op
        sps : dim of the local Hilbert space
        Green_op : a list of operators carrying momentum and that changes the sector of the GS
        Ouput : -Im Green function /pi
      """
    Psi0 = np.ones(basis.Ns, dtype=np.complex128) / np.sqrt(basis.Ns)
    E, Psi_GS = lanczos_gs(Psi0, H, M=M, nev=1)
    H, basis_q = bath_Hamil(L,N,U,mu,J,k = q,sps =sps)
    Green_op_Psi_GS = basis_q.Op_shift_sector(basis, Green_op, Psi_GS )
    Green_function = lanczos_green(w, Green_op_Psi_GS, H,E =E, M=M, Gamma=Gamma)
    return -Green_function.imag/np.pi

def S_qw( L,N,U,mu,J, Green_op_q, wvals, qvals, k =None,sps =N+1, M= 100, Gamma = 0.05):
    H,basis = bath_Hamil(L,N,U,mu,J,k = k,sps =sps)
    S_q_w = np.zeros( (len(qvals), len(wvals)), dtype = float)
    for i,q in enumerate(qvals):
        qint = int(round(L*q/(2*np.pi)))
        S_q_w[i,:] = full_procedure_per_qsector(H,basis,L,N,U,mu,J, Green_op_q[i],wvals,q=qint,k =k,sps = sps, M= M, Gamma = Gamma)
    return S_q_w

def full_procedure_polaron(bath_H,full_H,basis,L,N,U,g,mu,J,w,q,k =0,sps =N+1, M= 100, Gamma = 0.05):
    ''' ok it's annoying because i cannot optimize more, i cannot fix the overall momentum but only the individual ones like cannot define symmetries within tensored basis
    the information about the momentun of the impurity is hidden in full_H which is computed in the q momentum sector'''
    bath_Psi0 = np.ones(basis.Ns, dtype=np.complex128) / np.sqrt(basis.Ns) # initial state to look for bath GS
    E, bath_GS = lanczos_gs(bath_Psi0, bath_H, M=M, nev=1)

    #print("the GS ground state", E)

    ##Cq_imp_vac = np.exp(1j*np.arange(L)*q) / np.sqrt(L)
    ##Cq_imp_vac =  boson_basis_general(N=L, Nb=N, sps =sps, kxblock=(T, q))[0]
    Cq_imp_vac = np.array([1])
    full_Cq_GS = np.kron(bath_GS.ravel(), Cq_imp_vac)
    #print("shape of bath GS and of basis", bath_GS.shape, basis.Ns )
    ##full_Cq_GS = np.kron(bath_GS.ravel(), Cq_imp_vac)

    ##basis_q,full_H_q = polaron_Hamil(L,N,U,g,mu,J,k = q,sps =sps, q=q)
    #Green_function = lanczos_green(w- E, full_Cq_GS , full_H_q, M=M, Gamma=Gamma)
    Green_function = lanczos_green(w, full_Cq_GS , full_H, E=E, M=M, Gamma=Gamma)
    return -Green_function.imag/np.pi

def Polaron_spectral_fct(L,N,U,g,mu,J, wvals, qvals, k= None,sps = N+1, M= 100, Gamma = 0.05):
    bath_H, bath_basis = bath_Hamil(L,N,U,mu,J,k= k,sps=sps) # to construct the ground state of the bath
    ##full_basis, full_H = polaron_Hamil(L,N,U,g,mu,J,k = k,sps=N+1,q=i)
    A_q_w = np.zeros( (len(qvals), len(wvals)), dtype = float)
    for i,q in enumerate(qvals):
        
        qint = int(round(L*q/(2*np.pi)))
        _, full_H = polaron_Hamil(L,N,U,g,mu,J,k = k,sps=N+1,q=qint) #full hamiltonian living in the q momentum sector
        print('qint',qint)
        A_q_w[i,:] = full_procedure_polaron(bath_H,full_H,bath_basis,L,N,U,g,mu,J,wvals,q=q,k =k,sps = sps, M= M, Gamma = Gamma)
    return A_q_w

def modulo_2pi(k):
    """gives momentum into [-pi, pi)"""
    return ((k + np.pi) % (2*np.pi)) - np.pi
def TG_GS(N):
    # N chosen Odd
    Bethe_momenta = 2*np.pi/L*(-(N+1)/2 + np.arange(1,N+1))
    GS_energy = np.sum(free_spec(Bethe_momenta,mu) )
    return GS_energy, Bethe_momenta 
def TG_excitations(q,N):
    """particle-hole excitation of momentum q : hole has momentum k and particle k+q
    The spectrum is degenerate in Energy (many states for the same momenta)"""
    GS_energy, Bethe_momenta  = TG_GS(N)
    k_F = np.max(np.abs(Bethe_momenta))
    k_range = []
    for k in Bethe_momenta:
        if np.abs(k + q) >= k_F-0.0001:
            k_range.append(k)
    k_range = np.array(k_range)  
    GS_energy = 0 # bc I shifted the DSF already
    energies_q = GS_energy + free_spec(k_range+q,mu) - free_spec(k_range,mu)
    return energies_q

def plot_dyn_structure_factor(S_qw, q_vals, Omega_vals, Gamma=0.05):
    Q_mesh, Omega_mesh = np.meshgrid(q_vals, Omega_vals, indexing='ij')
    norm = mcolors.LogNorm(vmin=max(S_qw.min(), 1e-6), vmax=S_qw.max())
    plt.figure(figsize=(8,6))
    
    plt.pcolormesh(Q_mesh, Omega_mesh, S_qw, cmap='viridis', norm=norm, shading='auto')
    doublon_energies = [doublon_energy(U,q,J) for q in q_vals]
    plt.plot(q_vals, doublon_energies, c = 'g')
    
    #Tonks-Girardeau regime
    down_curve, up_curve = [np.min(TG_excitations(q,N)) for q in q_vals], [np.max(TG_excitations(q,N)) for q in q_vals]
    plt.plot(q_vals, down_curve, c='r')
    plt.plot(q_vals, up_curve, c='r')

    cbar = plt.colorbar(label= r"$S(q,\Omega)$")
    cbar.set_label(r"$S(q,\Omega)$", fontsize=14)
    x = np.linspace(-np.pi,np.pi,300)

    plt.xlabel("Momentum q", fontsize=14)
    plt.ylabel(r"Frequency $\Omega$", fontsize=14)
    plt.title("Dynamical Structure Factor"+rf", M={M+1}, $\Gamma$={Gamma}, U={U}, g={g}, L={L}, N={N}", fontsize=14)
    plt.tight_layout()
    plt.show()

start_t = time.time()
w =  np.linspace(-1, 10, 300)
bas = boson_basis_general(N=L, Nb=1, sps =2, kxblock=(T, 1))
print('taille base est ', bas.Ns)

Green_op_q = [[["n", [i], np.exp(-1j*q*i)/np.sqrt(L)] for i in range(L)]  for q in qvals]  #rho_q => DSF
S_q_w = S_qw(L,N,U,mu,J,Green_op_q,w,qvals,k=0,sps=N+1,M=M,Gamma=Gamma)
print('Computed DSF in ', time.time() - start_t,' seconds')
plot_dyn_structure_factor(S_q_w, qvals, w, Gamma)

start_t = time.time()
A_q_w = Polaron_spectral_fct(L,N,U,g,mu,J, w, qvals, k= 0,sps = N+1, M= M, Gamma = 0.05)
print('Computed polaron spectral fct in ', time.time() - start_t,' seconds')
plot_dyn_structure_factor(A_q_w, qvals, w, Gamma ) 


    