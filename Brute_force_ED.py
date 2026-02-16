from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import tensor_basis, boson_basis_1d # Hilbert space boson basis
from quspin.basis import boson_basis_general
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time
from quspin.basis import spin_basis_general
from quspin.operators import hamiltonian, quantum_LinearOperator
import scipy.sparse as sp
import numexpr, cProfile

L=10
a=1
N=4
n = N/L
J=1.0
U=0.0
mu=-2
g=-5.0
Gamma=0.05

# Translational symmetry
T = np.roll(np.arange(L), -1)


def free_spec(k,mu):
    return -2*np.cos(k*a) -mu
    #return k**2/(2*m_B) - mu   

def chem_potential(U,n):
    return -2 + U*n
def bogo_spec(k,U,n):
    Ek = free_spec(k ,chem_potential(U,n))
    return np.sqrt(abs((Ek + U*n)*(Ek +3*U*n)))

def Eg_vs_n(UN_list):
    Eg = []
    n = []
    hop=[[-J,i,(i+1)%L] for i in range(L)] #PBC
    for Uel in UN_list:
        print('UN element')
        U,N = Uel
        basis3  = boson_basis_1d(L, Nb=N,kblock = 0)
        interact=[[0.5*U,i,i] for i in range(L)] # U/2 \sum_j n_j n_j
        pot=[[-mu-0.5*U,i] for i in range(L)] # -(\mu + U/2) \sum_j j_n
        static=[['+-',hop],['-+',hop],['n',pot],['nn',interact]]
        H = hamiltonian(static, [], basis=basis3, dtype=np.complex128)
        E,V =  H.eigh()
        Eg.append(E[0])
        n.append(Uel[1]/L )
    plt.plot(n,Eg, color = 'r', label= 'Eg')
    #plt.plot(n, Un, color = 'b',label ='Un')
    plt.ylabel('Eg',size= 30)
    plt.xlabel('n',size=30)
    plt.legend(fontsize = 30)
    plt.title(f'L={L},Un={1/L}',size = 30)
    plt.show()

#UN_list = [[1/i,i] for i in range(1,8) ]
#Eg_vs_n(UN_list)


def Eg_vs_U(U_list):
    N2 = N//2
    U_list2 = np.copy(U_list)/2
    n2 = N2/L
    Eg = []
    Eg2 = []
    Un = [n*e for e in U_list]
    Un2 = [n2*e for e in U_list]

    hop=[[-J,i,(i+1)%L] for i in range(L)] #PBC

    basis3  = boson_basis_1d(L, Nb=N,kblock = 0)
    basis32  = boson_basis_1d(L, Nb=N2,kblock = 0)
    for Uel in U_list:
        print('U element')
        interact=[[0.5*Uel,i,i] for i in range(L)] # U/2 \sum_j n_j n_j
        pot=[[-mu-0.5*Uel,i] for i in range(L)] # -(\mu + U/2) \sum_j j_n
        static=[['+-',hop],['-+',hop],['n',pot],['nn',interact]]
        H = hamiltonian(static, [], basis=basis3, dtype=np.complex128)
        H2 = hamiltonian(static, [], basis=basis32, dtype=np.complex128)
        E,V =  H.eigh()
        E2,V2 =  H2.eigh()
        Eg.append(E[0])
        Eg2.append(E2[0])

    plt.plot(U_list/n,Eg, color = 'g', label= f'n={n}')
    plt.plot(U_list2/n2,Eg2, color = 'r', label= f'n={n2}')

    #plt.plot(U_list/n, Un, color = 'b',label ='Un')
    plt.ylabel('Eg',size= 30)
    plt.xlabel(r'$\gamma= U/n$',size=30)
    plt.legend(fontsize = 30)
    #plt.title(f'L={L},N={N}',size = 30)
    plt.title(f'L={L}',size = 30)
    plt.show()


##U_list = np.array([0.05*i for i in range(30) ])
##Eg_vs_U(U_list)

def Hamil(L,N,U,mu,J,k= None):
    #basis = boson_basis_1d(L, Nb=N, sps= N-2,kblock = k)
    basis = boson_basis_general(N=L, Nb=N) if k== None else  boson_basis_general(N=L, Nb=N, kxblock=(T, k))
    bath_hop=[[-J,i,(i+1)%L] for i in range(L)]
    bath_interact=[[0.5*U,i,i] for i in range(L)]
    bath_pot=[[-mu-0.5*U,i] for i in range(L)]
    bath_static = [['+-', bath_hop], ['-+', bath_hop], ['n', bath_pot], ['nn', bath_interact]]
    #H0 = quantum_LinearOperator(bath_static, basis=basis, dtype=np.complex128)
    H0 = hamiltonian(bath_static,[], basis=basis, dtype=np.complex128) 
    return H0,basis

# define custom LinearOperator object that generates the left hand side of the equation.
class LHS(sp.linalg.LinearOperator):
    #
    def __init__(self, H, omega, eta, E0, kwargs={}):
        self._H = H  # Hamiltonian
        self._z = omega + 1j * eta + E0  # complex energy
        self._kwargs = kwargs  # arguments
    @property
    def shape(self):
        return (self._H.Ns, self._H.Ns)
    @property
    def dtype(self):
        return np.dtype(self._H.dtype)
    def _matvec(self, v):
        # left multiplication
        return self._z * v - self._H.dot(v, **self._kwargs)
    def _rmatvec(self, v):
        # right multiplication
        return self._z.conj() * v - self._H.dot(v, **self._kwargs)
##### calculate action without constructing the Hamiltonian matrix

def dyn_structure_factor_ED(L,N,U,mu,J, q_vals, Omega_vals, Gamma=0.25):
    H0,basis0 = Hamil(L,N,U,mu,J,0)
    E0, psi0 = H0.eigsh(k=1, which='SA')
    E0 = E0[0]
    print("ground state energy:", E0, "and the norm of the GS is ", np.linalg.norm(psi0))
    S_qw = np.zeros((len(q_vals), len(Omega_vals)), dtype=float)
    for iq, q in enumerate(q_vals):
        print(f"momentum  q={q}")
        #basis_q = boson_basis_1d( L=basis0.L, Nb=basis0.Nb, sps=basis0.sps, kblock=q)
        qint = int(round(L*q/(2*np.pi)))
        Hq,basis_q = Hamil(L,N,U,mu,J,qint)
        rho_q_list = [["n", [i], np.exp(-1j*q*i)/np.sqrt(L)] for i in range(L)]
        #rho_q_list = [['n', [[np.exp(1j *q*j), j] for j in range(L)]]]
        
        psiA = basis_q.Op_shift_sector(basis0, rho_q_list, psi0)
        print("|psiA| =", np.linalg.norm(psiA))
        print('jusque la OK')
        # if A is zero then x too...
        if np.linalg.norm(psiA, ord=np.inf) < 1e-10:
            continue
        for iomega, omega in enumerate(Omega_vals):
            lhs = LHS(Hq, omega, Gamma, E0)
            x, *_ = sp.linalg.bicgstab(lhs, psiA, maxiter=1000)
            S_qw[iq, iomega] = -np.vdot(psiA, x).imag/np.pi
    
    print('min and max', np.min(S_qw), np.max(S_qw))
    return S_qw

#fast version with Op_shift_sector()
def dyn_structure_factor_ED_old2(q_vals, Omega_vals,L,N, U, mu, J, Gamma=0.25):
    H,basis = Hamil(L,N, 0,U,mu,J)
    E,V = H.eigh()
    [E0], psi0 = H0.eigsh(k=1, which="SA")

    print("ground state energy:", E0)
    num_q = len(q_vals)
    num_omega = len(Omega_vals)
    S_qw = np.zeros((num_q, num_omega))
    for iq, q in enumerate(q_vals):
        print('done with q =',q)
        #rho_q_list = [['n', [[np.exp(1j *q*j), j] for j in range(L)]]]
        #rho_q_list = [["n", [j,[np.exp(1j*q*j) ] for j in range(basis0.L)]]]
        rho_q_list = [["n", [i], np.exp(1j *q*i)] for i in range(L)]

        psiA = basis_q.Op_shift_sector(basis, rho_q_list, V[:, 0])
        basis_q = boson_basis_1d(L, Nb=N, sps=sps, kblock=q)  
        Hq = hamiltonian(bath_static, [], basis=basis_q, dtype=np.complex128)

        osc_strengths = np.abs(V.conj().T @ vec)**2
        for iomega, omega in enumerate(Omega_vals):
            denom = omega-E+1j*Gamma + E0
            S_qw[iq,iomega] = -np.sum((osc_strengths/denom).imag/np.pi)
    return S_qw



def dyn_structure_factor_ED1(L,N,U,mu,J, q_vals, Omega_vals, Gamma=0.25):
    H,basis =  Hamil(L,N,U,mu,J)
    num_q = len(q_vals)
    num_omega = len(Omega_vals)
    S_qw = np.zeros((num_q, num_omega))
    E,V = H.eigh()
    print("ground state energy:", E[0])
    for iq, q in enumerate(q_vals):
        print('done with q =',q)
        rho_q_list = [['n', [[np.exp(1j *q*j), j] for j in range(L)]]]
        rho_q = hamiltonian(rho_q_list, [], basis=basis, dtype=np.complex128, check_herm=False)
        vec = rho_q.dot(V[:, 0]) # rho q dot ground state
        osc_strengths = np.abs(V.conj().T @ vec)**2
        for iomega, omega in enumerate(Omega_vals):
            denom = omega-E+1j*Gamma + E[0]
            S_qw[iq,iomega] = -2*np.sum((osc_strengths/denom).imag)
    return S_qw


def plot_dyn_structure_factor(S_qw, q_vals, Omega_vals, Gamma=0.25):
    Q_mesh, Omega_mesh = np.meshgrid(q_vals, Omega_vals, indexing='ij')
    norm = mcolors.LogNorm(vmin=max(S_qw.min(), 1e-6), vmax=S_qw.max())
    plt.figure(figsize=(8,6))
    #plt.scatter(Q_mesh.flatten(), Omega_mesh.flatten(), c=S_qw.flatten(), cmap='viridis', norm=norm, s=40)
    plt.pcolormesh(Q_mesh, Omega_mesh, S_qw, cmap='viridis', norm=norm, shading='auto')

    cbar = plt.colorbar(label=r"$S(q,\Omega)$")
    cbar.set_label(r"$S(q,\Omega)$", fontsize=14)
    x = np.linspace(-np.pi,np.pi,300)
    plt.plot(x,bogo_spec(x,U,n),color = 'r')
    plt.xlabel("Momentum q", fontsize=14)
    plt.ylabel("Frequency Ω", fontsize=14)
    plt.title(f"Dynamical Structure Factor, Γ={Gamma}, U={U}, L={L}, N={N}", fontsize=14)
    plt.tight_layout()
    plt.show()


q_vals = np.array([2*np.pi/L*i - np.pi for i in range(L)])
Omega_vals = np.linspace(-4, 8, 55)


t = time.time()
##S_qw = dyn_structure_factor_ED(L,N,U,mu,J, q_vals, Omega_vals, Gamma)
##print('took (in seconds)', time.time() -t)
##plot_dyn_structure_factor(S_qw, q_vals, Omega_vals, Gamma)





bath_basis = boson_basis_1d(L, Nb=N,sps = 2)
print('basis',bath_basis)
bath_hop=[[-J,i,(i+1)%L] for i in range(L)]
bath_interact=[[0.5*U,i,i] for i in range(L)]
bath_pot=[[-mu-0.5*U,i] for i in range(L)]
bath_static = [['+-', bath_hop], ['-+', bath_hop], ['n', bath_pot], ['nn', bath_interact]]

H0 = hamiltonian(bath_static, [], basis=bath_basis, dtype=np.complex128, check_pcon=False, check_symm=False)


t = time.time()
#S_qw = dyn_structure_factor_ED(H0, bath_basis, q_vals, Omega_vals, Gamma)
#print('took (in seconds)', time.time() -t)
#plot_dyn_structure_factor(S_qw, q_vals, Omega_vals, Gamma)


imp_basis  = boson_basis_1d(L, Nb=1)


imp_hop=[[-J,i,(i+1)%L] for i in range(L)]
imp_pot=[[-mu,i] for i in range(L)]

bath_static1 = [['+-|', bath_hop], ['-+|', bath_hop], ['n|', bath_pot], ['nn|', bath_interact]]
imp_static = [['|+-', imp_hop], ['|-+', imp_hop], ['|n', imp_pot]]
imp_bath_int = [['n|n', [[g,j,j] for j in range(L)]]]
dynamic = []
static = bath_static1 + imp_static + imp_bath_int

basis1 = tensor_basis(bath_basis, imp_basis)

H1 = hamiltonian(static, [], basis=basis1, dtype=np.complex128, check_pcon=False, check_symm=False)


"""

def embed_bath_gs(gs_bath, imp_basis):
    imp_vac_vec = np.zeros(imp_basis.Ns, dtype=np.complex128)
    imp_vac_vec[0] = 1.0
    return np.kron(gs_bath, imp_vac_vec)


def impurity_creation_op(Q, L, basis_full):
    coeffs = [[np.exp(1j*Q*j)/np.sqrt(L), j] for j in range(L)]
    op_list = [['|+', coeffs]]   #acts only on impurity
    return hamiltonian(op_list, [], basis=basis_full, dtype=np.complex128,check_herm=False, check_symm=False, check_pcon=False)
"""

def impurity_Q(Q,L):
    E0, V0 = H0.eigh()
    gs_bath = V0[:,0]
    Egs = E0[0]
    imp_Q = np.exp(1j*np.arange(imp_basis.Ns) * Q) / np.sqrt(L)
    vec_full = np.kron(gs_bath, imp_Q)
    return vec_full, Egs


def spectral_function_ED(Q_vals, Omega_vals, eta=0.2):
    t0 = time.time()
    E1, V1 = H1.eigh()
    print('Energies', E1)
    A = np.zeros((len(Q_vals), len(Omega_vals)), dtype=float)
    for iq, Q in enumerate(Q_vals):
        vec_excited,Egs = impurity_Q(Q,L)
        overlaps = np.abs(V1.conj().T @ vec_excited)**2
        print('overlaps',overlaps)
        for iw, Omega in enumerate(Omega_vals):
            denom = Omega - (E1 - Egs) + 1j*eta
            A[iq, iw] = -2.0 * np.sum(overlaps / denom).imag

    print("spectral computed in {:.2f}s".format(time.time()-t0))
    return A


def plot_spectral_function(Q_vals, Omega_vals, A, cmap="viridis", vmin=1e-8, vmax=None, figsize=(8,5)):
    Q_mesh, O_mesh = np.meshgrid(Q_vals, Omega_vals, indexing='ij')
    A_safe = np.copy(A)
    vmax = A_safe.max()
    vmin=max(vmin, A_safe.min()+1e-12)
    print('vmax is',vmax, 'and vmin is', vmin)
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    plt.figure(figsize=figsize)
    #plt.scatter(Q_mesh.flatten(), O_mesh.flatten(), c=A.flatten(), cmap=cmap, norm=norm, s=12)
    plt.pcolormesh(Q_mesh, O_mesh, A, cmap='viridis', norm=norm, shading='auto')
    plt.colorbar(label=r'$\mathcal{A}(Q,\Omega)$')
    plt.xlabel('Q')
    plt.ylabel(r'$\Omega$')
    plt.title(fr'Spectral function $A(Q,\Omega), g={g},L={L}, U={U}, N={N}, \Gamma={Gamma}$')
    plt.tight_layout()
    plt.show()


Q_vals = np.linspace(-np.pi, np.pi, L)
Omega_vals = np.linspace(-5, 12, 200)
Omega_vals = np.linspace(-10,10,10*L)

A = spectral_function_ED(Q_vals, Omega_vals, eta=Gamma)
#print('min and max', np.min(A), np.max(A))
print('took (in seconds)', time.time() -t)
plot_spectral_function(Q_vals, Omega_vals, A)



#plot_spectral_function(Q_vals, Omega_vals, A, title=f"L={L}, Nbath={N}, g={g}")




