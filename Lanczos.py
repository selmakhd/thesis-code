from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import tensor_basis, boson_basis_1d, boson_basis_general # Hilbert space boson basis
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import time
from scipy.sparse.linalg import eigsh
from scipy.io import mmwrite
from scipy.sparse import csr_matrix
from Lanczos_algo import *

#Here I am looking at DSF, Polaron with many different potentials (sine, contact, Coulomb, dipolar)
#todo :  DSF vs g, TN coeff, Trace thing


a=1
J=1.0
mu=-2
#V0 = 1.2  #ext pot amplitude for the sine potential
#Ext_pot_q = 2*np.pi/3
Ext_pot_q = -np.pi
Ext_pot_phi = 0
#Vc = 0 # ext pot amplitude for the power law potential
#alpha = 3  #the power of the power law potential
Yukawa_lam = 1
Gamma=0.05
m_B = 1



def free_spec(k,mu):
    return -2*np.cos(k*a) -mu
    #return k**2/(2*m_B) - mu   

def chem_potential(U,n):
    return -2 + U*n
def bogo_spec(k,U,n):
    Ek = free_spec(k ,chem_potential(U,n))
    return np.sqrt(abs((Ek + U*n)*(Ek +3*U*n)))
def doublon_energy(U, k,J):
    E_trial = np.linspace(4,11,500)
    LS = np.array([1/L*sum(1/(E -4+4*J*np.cos(k/2)*np.cos(qvals))) for E in E_trial ])
    best_index = np.argmin( np.abs(LS -1/U)  )
    print('for momentum k ', k)
    print('best fitting energy and the difference',E_trial[best_index], np.abs(LS -1/U)[best_index] )
    return E_trial[best_index]
    #return 4 + np.sqrt(U*U + 4*J*np.cos(k/2))

def bath_Hamil(L,N,U,U2,Vc,V0,alpha,mu,J,T,k= None,sps = None, Ext_pot_q=0):
    """ Gives me the bath hamiltonian and the basis within the specified momentum sector k """
    if sps is None:
        sps = N + 1
    basis = boson_basis_general(N=L, Nb=N, sps = sps) if k== None else  boson_basis_general(N=L, Nb=N, sps =sps, kxblock=(T, k))
    bath_hop=[[-J,i,(i+1)%L] for i in range(L)]
    bath_interact=[[0.5*U,i,i] for i in range(L)]
    bath_interact2=[[0.5*U2,i,(i+1)%L] for i in range(L)]
    bath_pot=[[-mu-0.5*U + V0*np.cos(i*Ext_pot_q + Ext_pot_phi),i] for i in range(L)]
    if Vc != 0.0:
        for i in range(L):
            for j in range(i+1, L):
                d = min(abs(i - j), L - abs(i - j))
                V = Vc/d**alpha
                #V= Vc/d*np.exp(-d/Yukawa_lam) #Yukawa interaction
                bath_interact.append([V, i, j])
    bath_static = [['+-', bath_hop], ['-+', bath_hop], ['n', bath_pot], ['nn', bath_interact], ['nn', bath_interact2]]
    H0 = hamiltonian(bath_static,[], basis=basis, dtype=np.complex128) 
    return H0,basis

def polaron_Hamil(L,N,U,U2,Vc,V0,alpha,g,mu,J,T,bias,k= None,sps = None):
    """ Gives me the hamiltonian of the bath + impurity and the basis within the specified momentum sector k 
    Remark : i add a small term to lift the degeneracy of momentum"""
    if sps == None:
        sps = N + 1
    print('in polaron hamil sps is ', sps)
    bath_basis = boson_basis_general(N=L, Nb=N, sps = sps) if k== None else  boson_basis_general(N=L, Nb=N, sps =sps, kxblock=(T, k))
    imp_basis  = boson_basis_general(N=L, Nb=1,sps =2)
    print("len of bath basis and len of imp basis ", bath_basis.Ns, imp_basis.Ns)
    tensored_basis = tensor_basis(bath_basis, imp_basis)

    hop=[[-J,i,(i+1)%L] for i in range(L)]
    bath_interact=[[0.5*U,i,i] for i in range(L)]
    bath_interact2=[[0.5*U2,i,(i+1)%L] for i in range(L)]
    bath_pot=[[-mu-0.5*U   + V0*np.cos(i*Ext_pot_q + Ext_pot_phi)    ,i] for i in range(L)]
    if Vc != 0.0:
        for i in range(L):
            for j in range(i+1, L):
                d = min(abs(i-j), L - abs(i - j))
                V = Vc/d**alpha
                #V= Vc/d*np.exp(-d/Yukawa_lam) #Yukawa interaction
                bath_interact.append([V, i, j])
    imp_chem=[[-mu,i] for i in range(L)]

    bath_static = [['+-|', hop], ['-+|', hop], ['n|', bath_pot], ['nn|', bath_interact], ['nn|', bath_interact2]]

    imp_static = [['|+-', hop], ['|-+', hop], ['|n', imp_chem]]
    eps = -0.0000000001#lift degeneracy
    imp_momentum_plus = [[ 1j*eps, i, (i+1) % L] for i in range(L)]
    imp_momentum_minus = [[-1j*eps, i, (i+1) % L] for i in range(L)]
    imp_static += [['|+-', imp_momentum_plus], ['|-+', imp_momentum_minus]]
    eps =  bias*0.0000000001
    bos_momentum_plus = [[ 1j*eps, i, (i+1) % L] for i in range(L)]
    bos_momentum_minus = [[-1j*eps, i, (i+1) % L] for i in range(L)]
    bath_static += [['+-|', bos_momentum_plus], ['-+|', bos_momentum_minus]]

    imp_bath_int = [['n|n', [[g,j,j] for j in range(L)]]]

    static = bath_static + imp_static + imp_bath_int

    polaron_H = hamiltonian(static, [], basis=tensored_basis, dtype=np.complex128, check_pcon=False, check_symm=False)

    return tensored_basis, polaron_H


def full_procedure_per_qsector(H,basis,L,N,U,U2,Vc,V0,alpha,mu,J, Green_op,w,T,q,k =0,sps =None, M= 100, Gamma = 0.05, Ext_pot_q=0):
    """  I have to run the Lanczos algorithm once  with some intial state in the right sector
        to get the ground state. Then I rerun it again with an intial state psi0 = hat{a}|GS>
        Input
        H,basis : hamiltonian and basis of the bosons in the q sector 
        L,N,U,mu,J : system parameters
        k : momentum sector for the GS
        q : momentum carried by Green_op
        sps : dim of the local Hilbert space
        Green_op I assume it is momentum definite operator that changes the sector of the GS

        Ouput : -Im Green function /pi
      """
    if sps is None:
        sps = N + 1
    #print("enter fct")
    Psi0 = np.ones(basis.Ns, dtype=np.complex128) / np.sqrt(basis.Ns)
    print("step1")
    E, Psi_GS = lanczos_gs(Psi0, H, M=M, nev=1)
    print("step2")
    #print("GS energy is ", E, "and the GS norm is ", norm(Psi_GS))
    ##H, basis_q = bath_Hamil(L,N,U,U2,Vc,alpha,mu,J,T,k = q,sps =sps,Ext_pot_q=Ext_pot_q)

    #basis_q, H = polaron_Hamil(L,N,U,U2,Vc,alpha,g,mu,J,T,k= k,sps = sps)
    ##Green_op_Psi_GS = basis_q.Op_shift_sector(basis, Green_op, Psi_GS )
    Green_op_matrix =  hamiltonian(Green_op,[], basis=basis, dtype=np.complex128, check_herm=False)  #carries info about q
    Green_op_Psi_GS =   Green_op_matrix.dot(np.array(Psi_GS))
    #print("norm psi A is ", norm(Green_op_Psi_GS))
    Green_function = lanczos_green(w, Green_op_Psi_GS, H,E =E, M=M, Gamma=Gamma)
    return -Green_function.imag/np.pi

def S_qw( L,N,U,U2,Vc,V0,alpha,mu,J, Green_op_q, wvals, qvals,T, k =None,sps =None, M= 100, Gamma = 0.05, Ext_pot_q=0):
    if sps is None:
        sps = N + 1
    H,basis = bath_Hamil(L,N,U,U2,Vc,V0,alpha,mu,J,T,k = k,sps =sps, Ext_pot_q = Ext_pot_q)
    #print('basis length', basis.Ns)
    S_q_w = np.zeros( (len(qvals), len(wvals)), dtype = float)
    for i,q in enumerate(qvals):
    #for i,q in enumerate([qvals[0]]):

        qint = int(round(L*q/(2*np.pi)))
        print('qint',qint)
        #print("hamil", H)
        S_q_w[i,:] = full_procedure_per_qsector(H,basis,L,N,U,U2,Vc,V0,alpha,mu,J, Green_op_q[i],wvals,T,q=qint,k =k,sps = sps, M= M, Gamma = Gamma, Ext_pot_q=Ext_pot_q)
    #print('min and max', np.min(S_q_w), np.max(S_q_w))
    return S_q_w

    

def full_procedure_polaron(bath_H,full_H,basis,L,N,U,U2,Vc,V0,g,mu,J,w,q,k =0,sps =None, M= 100, Gamma = 0.05):
    if sps is None:
        sps = N + 1
    bath_Psi0 = np.ones(basis.Ns, dtype=np.complex128) / np.sqrt(basis.Ns) # initial state to look for bath GS
    E, bath_GS = lanczos_gs(bath_Psi0, bath_H, M=M, nev=1)
    #print("the GS ground state", E)
    Cq_imp_vac = np.exp(1j*np.arange(L)*q) / np.sqrt(L)
    #print("shape of bath GS and of basis", bath_GS.shape, basis.Ns )
    full_Cq_GS = np.kron(bath_GS.ravel(), Cq_imp_vac)                              
    print('the norm of the intial vectors is', norm(bath_Psi0), norm(full_Cq_GS) )
    #basis_q,full_H_q = polaron_Hamil(L,N,U,U2,Vc,alpha,g,mu,J,T,k = q,sps =sps)
    #Green_function = lanczos_green(w- E, full_Cq_GS , full_H_q, M=M, Gamma=Gamma)
    print('before Green')
    #print('full Cq GS',  full_Cq_GS.shape)
    #print('full H',  full_H.shape)
    Green_function = lanczos_green(w, full_Cq_GS , full_H, E=E, M=M, Gamma=Gamma)
    print('after Green')
    #print("step 6")
    return -Green_function.imag/np.pi

def Polaron_spectral_fct(L,N,U,U2,Vc,V0,alpha,g,mu,J, wvals, qvals,T,bias, k= None,sps = None, M= 100, Gamma = 0.05, Ext_pot_q=Ext_pot_q):
    if sps is None:
        sps = N + 1
    print('sps is', sps)
    
    
    A_q_w = np.zeros( (len(qvals), len(wvals)), dtype = float)
    for i,q in enumerate(qvals): # attention remet bath hamil et polaron hamil avant la boucle sinon c pour rien c indep de q
        qint = int(round(L*q/(2*np.pi)))
        print('qint',qint)
        print('bath arguments', L,N,U,U2,Vc,V0,alpha,mu,J,T,k,sps , Ext_pot_q )
        bath_H, bath_basis = bath_Hamil(L,N,U,U2,Vc,V0,alpha,mu,J,T,k= k,sps = sps, Ext_pot_q=Ext_pot_q) 
        print('polaron H arguments', L,N,U,U2,Vc,V0,alpha,g,mu,J,T,1,k ,sps)
        full_basis, full_H = polaron_Hamil(L,N,U,U2,Vc,V0,alpha,g,mu,J,T,bias,k = k,sps =sps)
        print("about to compute polaron spec for given q and Vc", q,Vc)
        print('the arguments are ', L,N,U,U2,Vc,V0,g,mu,J, q, k , sps,M, Gamma )
        A_q_w[i,:] = full_procedure_polaron(bath_H,full_H,bath_basis,L,N,U,U2,Vc,V0,g,mu,J,wvals,q=q,k =k,sps = sps, M= M, Gamma = Gamma)
        total_weight = np.trapz(A_q_w[i,:], wvals)   # integrate over omega
        print(f"q = {q:.3f}, total spectral weight = {total_weight:.5f}")
    return A_q_w

def modulo_2pi(k):
    """gives momentum into [-pi, pi)"""
    return ((k + np.pi) % (2*np.pi)) - np.pi

def TG_GS(N,L):
    # This formula works both for N odd and even
    Bethe_momenta = 2*np.pi/L*(-(N+1)/2 + np.arange(1,N+1))
    GS_energy = np.sum(free_spec(Bethe_momenta,mu) )
    return GS_energy, Bethe_momenta 

def TG_excitations(q,N,L):
    """particle-hole excitation of momentum q : hole has momentum k and particle k+q
    The spectrum is degenerate in Energy (many states for the same momentum)"""
    GS_energy, Bethe_momenta  = TG_GS(N,L)
    k_F = np.max(np.abs(Bethe_momenta))
    k_range = []
    for k in Bethe_momenta:
        if np.abs(k + q) > k_F:
            k_range.append(k)
    k_range = np.array(k_range)  
    GS_energy = 0 # bc I shift the DSF
    if len(k_range) == 0:
        energies_q = GS_energy 
    else:
        energies_q = GS_energy + free_spec(k_range+q,mu) - free_spec(k_range,mu)
    #print('k_range', k_range)
    return energies_q
def test_Fermiplot(qvals):
    down_curve, up_curve = [np.min(TG_excitations(q,N,L)) for q in qvals], [np.max(TG_excitations(q,N,L)) for q in qvals]
    plt.plot(qvals, down_curve, c='r')
    plt.plot(qvals, up_curve, c='r')
    plt.xlabel("Momentum q", fontsize=14)
    plt.ylabel("Energy", fontsize=14)
    #plt.title(obs+f", M={M+1}, Γ={Gamma}, U={U}, L={L}, N={N}", fontsize=14)
    plt.show()


def doublon_energy(q, U, J):
    return 4 + np.sqrt(U*U + (4*J*np.cos(q/2))**2)

def two_plus_one_continuum(K, qvals, U, J):
    
    return doublon_energy(K - qvals, U, J) + free_spec(qvals, -2)
    #return doublon_energy(K - qvals, U, J) + free_spec(qvals, 0)
    #return  free_spec(K - qvals, -2)+ free_spec(qvals, -2)
    #return doublon_energy(K + qvals, U, J) - free_spec(qvals, mu)



def plot_dyn_structure_factor(L,N,U,U2,Vc,V0,alpha,g,M,S_qw, q_vals, Omega_vals, Gamma=0.05, yaxis = r"$S(q,\Omega)$", obs = "Dynamical Structure Factor",Ext_pot_q=0, bound = None,sps = None, xaxis = r"momentum q"):
    print('heres, sqw shape', np.shape(S_qw))
    Q_mesh, Omega_mesh = np.meshgrid(q_vals, Omega_vals, indexing='ij')
    print('heres, Q_mesh, Omega_mesh ', np.shape(Q_mesh), np.shape(Omega_mesh))
    norm = mcolors.LogNorm(vmin=max(S_qw.min(), 1e-6), vmax=S_qw.max())
    #norm = mcolors.Normalize(vmin=S_qw.min(), vmax=S_qw.max())
    plt.figure(figsize=(8,6))
    #plt.scatter(Q_mesh.flatten(), Omega_mesh.flatten(), c=S_qw.flatten(), cmap='viridis', norm=norm, s=40)
    #Tonks-Girardeau regime
    
    if bound == None:
        three_body_up = np.array([np.max(two_plus_one_continuum(K, q_vals, U, J) )  for K in q_vals])
        three_body_down = np.array([np.min(two_plus_one_continuum(K, q_vals, U, J) )  for K in q_vals])
        E =[U/np.abs(U)*np.sqrt(U**2 + (4*J*np.cos(K/2))**2) +4 for K in q_vals]
        #doublon_energies = [doublon_energy(U,q,J) for q in q_vals]
        down_curve, up_curve = [np.min(TG_excitations(q,N,L)) for q in q_vals], [np.max(TG_excitations(q,N,L)) for q in q_vals]
        #print('up curve', up_curve)
        #print('down curve', down_curve)
        plt.plot(q_vals, down_curve, c='r') 
        plt.plot(q_vals, up_curve, c='r') 
        plt.plot(q_vals, three_body_up , c='g') 
        plt.plot(q_vals, three_body_down , c='g') 
        plt.plot(q_vals, E)
        #plt.plot(q_vals, doublon_energies, c = 'g')
        #S_qw[L//2,:] = np.zeros( len(Omega_vals), dtype = S_qw.dtype ) + 1e-6 #if i want to include the connected part only 
    ##plt.plot(q_vals, free_spec(q_vals,mu),linestyle='--', color='magenta')
    print('heres, sqw shape', np.shape(S_qw))
    plt.pcolormesh(Q_mesh, Omega_mesh, S_qw, cmap='viridis', norm=norm, shading='auto')
    
    cbar = plt.colorbar(label=yaxis)
    cbar.set_label(yaxis, fontsize=14)
    x = np.linspace(-np.pi,np.pi,300)
    
    #plt.plot(x,bogo_spec(x,U,n),color = 'r')
    plt.xlabel(xaxis, fontsize=14)
    plt.ylabel(r"Frequency $\Omega$", fontsize=14)
    #plt.title(obs+f", M={M+1}, Γ={Gamma}, U={U}, g ={g}, L={L}, N={N}", fontsize=14)
    #plt.title(obs+rf", M={M+1}, $\Gamma$={Gamma}, U={U}, g={g}, L={L}, N={N}, q = {round(Ext_pot_q,3)}, V={V0}", fontsize=14)
    plt.title(obs+rf", M={M+1}, $\Gamma$={Gamma}, U={U}, U2={U2}, Vc={round(Vc,3)}, V0={V0}, $\alpha=${alpha}, g={g}, L={L}, N={N}, sps={sps}", fontsize=14)
    
    #plt.title(obs+rf", M={M+1}, $\Gamma$={Gamma}, U={U}, g={g}, L={L}, N={N}, q = {round(Ext_pot_q,3)}, V={Vc}", fontsize=14)
    
    plt.tight_layout()
    plt.show()


#H_test,basis_test = bath_Hamil(10,3,0.2,U2,Vc,alpha,mu,J,T,k= None,Ext_pot_q =Ext_pot_q)
#print('basis len',basis_test.Ns)

#np.random.seed(42)
#A = np.random.rand(5,5)
#A = A.T + A 
#print('A=',A)
#v= np.random.rand(5)
#print('v=',v)
#lanczos_gs(v, A, M=2, nev=1)
#print(lanczos_gs(v, A, M=2, nev=1))




# To test with Julia....
#H,basis = bath_Hamil(L,N,U,U2,Vc,alpha,mu,J,T,k = 0,sps =None, Ext_pot_q=Ext_pot_q)
#mmwrite("hamiltonian.mtx", H.tocsc())


##Green_op_q = [[["n", [i], np.exp(-1j*q*i)/np.sqrt(L)] for i in range(L)]  for q in qvals]  #rho_q => DSF



    