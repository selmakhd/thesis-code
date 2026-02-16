from Lanczos import *
from find_polaron_wavefunction import *
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh


L = 10
N = 2
n = N/L
J=1
mu = -2
U = 2 #Contact
U2 = 0 #next neighbor interaction
V0 =0 #external cosine potential
Vc = 100/n**3 # ext pot amplitude for the power law potential
Vc=0
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

def n_pow(i,sps):
    if sps == None:
        sps = N+1 
    basis = boson_basis_general(N=1, sps = sps)
    ni = [['n', [[1,0]]]]
    static = ni 
    ni_op = hamiltonian(static, [], basis=basis, dtype=np.complex128, check_pcon=False, check_symm=False)
    print('the basis is', basis)
    op = ni_op.copy().toarray()
    powers= np.zeros(i+1,dtype=object)
    powers[0] = op 
    for j in range(i-1):
        op = op @ ni_op.toarray()
        powers[j+1] = op
    return powers
        

# reduced single site density matrix
def SSRDM(w_window,K,N,U,U2,Vc,V0,alpha,g,mu,J,T,L = L, tolerance =0.2,sps = None,M=M):
    """ Single site reduced density matrix (maybe do it for the traced out)
    if i trace out the bosons dof i would basically have the polaron schmidt decomposition but at a single site...
    ok so in the local hilbert space the first vector is the highest occupation
    """
    if sps == None:
        sps = N+1
    
    Psi_K,psiw,psik,k_int,full_basis,Z  = Psi_polaron(w_window,K,N,U,U2,Vc,V0,alpha,g,mu,J,T,L = L,tolerance = tolerance, sps =sps,M=M)
    basis_b = boson_basis_general(N=L, Nb=N, sps = sps)
    #i trace out and keepong the sub system A = left = bosons
    rho_bosons = full_basis.partial_trace(Psi_K, sub_sys_A="left",return_rdm='A', enforce_pure=True, sparse=False)
    print('shape rho_bosons and type',rho_bosons .shape, type(rho_bosons .shape))
    print('rho bosons looks like this',rho_bosons)
    #i trace out the sub system B = non zero site
    rho_site = basis_b.partial_trace(rho_bosons, sub_sys_A=[0], subsys_ordering=False, return_rdm='A', enforce_pure=False, sparse=False)
    print('rho boson',rho_site )
    print('shape boson and type',rho_site .shape, type(rho_site .shape))
    #lambdas = np.zeros(sps,dtype = float)
    #for i in range(sps):
    #    lambdas[i] = np.abs(rho_site[0][i][i])

    #print("occupation proba", lambdas)
    #n_op = n_pow(sps -1,sps)
    #n1 = np.trace(n_op[0]@rho_site[0])
    #n2 = np.trace(n_op[1]@rho_site[0])
    #lam2 =(n2-n1)/2
    #lam1 = n1 -2*lam2
    #print(' lambda2 is',lam2   )
    #print(' lambda1 is',lam1 )
    #print(' lambda0 is',1-lam1-lam2 )
    rho = rho_site[0].real  # 2D array
    diag = np.diag(rho)
    for idx, val in enumerate(diag):
        print(f"index {idx} -> P(n={sps-idx-1}) = {val:.6g}")
    print("sum diag:", diag.sum())
    #print('n1 and n2', n1,n2)

    return diag

def RDM_GS(N,U,U2,Vc,V0,alpha,g,mu,J,T,L = L,sps = None,M=M):
    if sps == None:
        sps = N+1
    print('step1')
    H_bath,basis_bath =bath_Hamil(L,N,U,U2,Vc,V0,alpha,mu,J,T,k= None,sps = sps, Ext_pot_q=0)
    print('step2')
    #H_sparse = csr_matrix(H_bath)  # convert dense to sparse
    E, Vec = H_bath.eigh()
    V = Vec[:,0]
    print('step3')

    #E,V = eigsh(H_bath, k=1, which='SA', return_eigenvectors=True)
    rho_bosons = basis_bath.partial_trace(V,sub_sys_A=[0], subsys_ordering=False, return_rdm='A', enforce_pure=False, sparse=False)
    print('step4')
    return rho_bosons 

#print(RDM_GS(N,U,U2,g,mu,J,T,L = L,sps = None,M=M))
#for i in range(9999999999999999999999):
#    b=0
w_window = (-6,-2.5)
w_window = (-3,0)
#w_window = (-10,-7.5)
w_window = (-10,-5)
w_window = (-2.5,0)
w_window = (-0.5,2.5)
w_window = (-3.5,-2)
w_window = (0,2)
w_window = (-5.5,-3)
w_window = (-10,-7.5)
w_window = (-7.5,-4)
w_window = (-1,1)
w_window = (2.5,5)
w_window = (0,2.5)
w_window = (4.7,5)
w_window = (0,2.5)
w_window = (-2.5,-0.5)
w_window = (-5,-2.6)
K =2*np.pi/L
K= -np.pi
K=0


#SSRDM(w_window,K,N,U,U2,g,mu,J,T,L = L, tolerance =0.2,sps = sps,M=M)
#for i in range(100000000000000):
#    b=0

# impurity boson correlation fct
def imp_bos_corr(w_window,K,N,U,U2,Vc,V0,alpha,g,mu,J,T,L = L, tolerance =0.2,sps = None,M=M):
    if sps == None:
        sps = N+1
    
    Psi_K,psiw,psik,k_int,full_basis,Z  = Psi_polaron(w_window,K,N,U,U2,Vc,V0,alpha,g,mu,J,T,L = L,tolerance = tolerance, sps =sps,M=M)
    basis_b = boson_basis_general(N=L, Nb=N, sps = sps)

    #imp_basis  = boson_basis_general(N=L, Nb=1,sps =2)
    corr = np.zeros(L,dtype = complex)
    for j in range(L):
        #I assume translation symmetry
        imp_bath_corrj = [['n|n', [[1,0,j]]]]
        static =imp_bath_corrj
        corr_j_op= hamiltonian(static, [], basis=full_basis, dtype=np.complex128, check_pcon=False, check_symm=False)
        #corr[j] = Psi_K.conj().T @ corr_j_op @ Psi_K
        corr[j] = corr_j_op.expt_value(Psi_K)
        #print('corrj is ',corr[j])
    print('ok done')
    return corr


#two_body_corr=imp_bos_corr(w_window,K,N,U,U2,Vc,V0,alpha,g,mu,J,T,L = L, tolerance =0.2,sps = 2,M=M)

def two_body_corr_plot(two_body_corr):
    plt.figure(figsize=(6,4))
    plt.plot(np.arange(L), np.real(two_body_corr), 'o-', label=r'Re[$\langle n_\mathrm{imp} n_j\rangle$]')
    plt.xlabel('Bath site j')
    plt.ylabel('Correlation')
    plt.title('Impurity-Boson Two-Body Correlation')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

#two_body_corr_plot(two_body_corr)

# impurity boson-boson correlation fct
def three_body_corr(w_window,K,N,U,U2,Vc,V0,alpha,g,mu,J,T,L = L, tolerance =0.2,sps = None,M=M):
    if sps == None:
        sps = N+1
    
    Psi_K,psiw,psik,k_int,full_basis,Z  = Psi_polaron(w_window,K,N,U,U2,Vc,V0,alpha,g,mu,J,T,L = L,tolerance = tolerance, sps =sps,M=M)
    test_H0,test_basis = bath_Hamil(L,N,U,U2,Vc,V0,alpha,mu,J,T,k= None,sps = sps, Ext_pot_q=0)
    _,test_GS = test_H0.eigh()
    print('psi K', Psi_K)
    #for i in range(100000000000000000000000000):
    #    b=1
    ##basis_b = boson_basis_general(N=L, Nb=N, sps = sps)
    corr = np.zeros((L,L),dtype = complex)
    test_list =[]
    for j in range(L):
        for i in range(L):
            #I assume translation symmetry
            imp_bath_corrji = [['++--|n', [[1,i,j,i,j,0]]]]
            #imp_bath_corrji = [['|nn', [[1,i,j]]]]
            #imp_bath_corrji = [['|nnn', [[1,i,j,0]]]]
            static =imp_bath_corrji
            corr_ji_op= hamiltonian(static, [], basis=full_basis, dtype=np.complex128, check_pcon=False, check_symm=False)
            corr[j,i] = corr_ji_op.expt_value(Psi_K)

            ##test_bath_corr=[[1,i,j]]
            ##test_bath_static = [ ['nn', test_bath_corr]]
            ##test_corr_ji_op= hamiltonian(test_bath_static, [], basis=test_basis, dtype=np.complex128, check_pcon=False, check_symm=False)
            ##test_state =test_GS[:,0]
            #corr[j,i] = test_corr_ji_op.expt_value(test_state)

            ##print('the GS is ', test_state)

        ##test_bath_corr2=[[1,j]]+[[0,k] for k in range(L) if k!=j]
        ##test_bath_static2 = [ ['n', test_bath_corr2]]
        ##test_bath_static2 = [['n', [[1, j]]]]  
        ##test_corr_j_op2= hamiltonian(test_bath_static2, [], basis=test_basis, dtype=np.complex128, check_pcon=False, check_symm=False)
        ##test_state =test_GS[:,0]
        #test_state = np.ones(len(test_state))/np.sqrt(len(test_state))

        ##test_list.append(test_corr_j_op2.expt_value(test_state))
        print('ok done')
    ##print('corr is', corr)
    ##print('density', test_list)
    ##print('test state', test_state)
    ##print("list coeff", test_bath_corr2)
    return corr, psiw,k_int,Z

imp_bos_bos,psiw,k_int,Z=three_body_corr(w_window,K,N,U,U2,Vc,V0,alpha,g,mu,J,T,L = L, tolerance =0.2,sps = sps,M=M)
##imp_bos_bos,psiw,k_int,Z=three_body_corr2(w_window,K,N,U,U2,Vc,V0,alpha,g,mu,J,T,L = L, tolerance =0.2,sps = sps,M=M) # make sure you plot it right though

#imp_bos_bos2=three_body_corr2(w_window,K,N,U,U2,Vc,V0,alpha,g,mu,J,T,L = L, tolerance =0.2,sps = None,M=M)
#two_body_corr_plot(imp_bos_bos2)

def plot_three_body_corr(corr,psiw,k_int,Z, num_levels=50, gamma=0.5, title='Three-body correlation'):
    """
    Plot correlation matrix with dot size and color encoding the correlation magnitude.

    corr : 2D np.array, shape (L,L) Correlation matrix
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

    plt.colorbar(sc, label=r'Re(<n_i n_j>)')
    plt.xlabel('Site i')
    plt.ylabel('Site j')
    plt.title(title + fr'E={round(psiw,3)}, k_int={k_int}, Z={round(Z,4)}, N={N}, L={L}, U={U}, U2={U2}, Vc={round(Vc,2)}, $\alpha=${alpha}, g={g}, sps={sps}' )
    plt.xticks(np.arange(L))
    plt.yticks(np.arange(L))
    plt.gca().set_aspect('equal')
    plt.show()

plot_three_body_corr(imp_bos_bos,psiw,k_int,Z, num_levels=50, gamma=0.5, title='Three-body correlation')



