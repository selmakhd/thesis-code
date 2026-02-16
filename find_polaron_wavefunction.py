from scipy.sparse.linalg import eigsh
from Lanczos import *

def polaron_momentum(psi,full_basis,L,N,M= 100,Gamma = 0.05, Ext_pot_q=0):
    '''to compute momentum i want to generalize to the many-body
     case the arg(psi(0)/psi(1)) trick by defining "all-zero" and "all-one" configurations 
     note that this only works when the amplitude is not zero for the state where all the boson stay at the same site i'''
    bath0_str = "1" * N + "0" * (L-N) # better to deal with hardcore bosons 
    bath1_str = "0" + "1" * N + "0" * (L-N-1)
    imp0_str = "1" + "0" * (L-1)
    imp1_str = "0" + "1" +  "0" * (L-2)
    #integer conversion
    bath0_int = full_basis.basis_left.state_to_int(bath0_str)
    imp0_int  = full_basis.basis_right.state_to_int(imp0_str)
    bath1_int = full_basis.basis_left.state_to_int(bath1_str)
    imp1_int  = full_basis.basis_right.state_to_int(imp1_str)
    #subbase indices
    bath_index0 = full_basis.basis_left.index(bath0_int)
    imp_index0  = full_basis.basis_right.index(imp0_int)
    bath_index1 = full_basis.basis_left.index(bath1_int)
    imp_index1  = full_basis.basis_right.index(imp1_int)
    #flattened indices
    all_zero_space_index = bath_index0*L+ imp_index0
    all_one_space_index  = bath_index1*L + imp_index1
    #compute phase
    psi0 = psi[all_zero_space_index]
    psi1 =psi[all_one_space_index]
    norm_psi0 = np.abs(psi0)
    norm_psi1 =np.abs(psi1)
    check_norm = (norm_psi1 ,norm_psi0)
    if norm_psi0 < 1e-16 or norm_psi1 < 1e-16:
        print('psi at first site is',norm_psi0,norm_psi1 )
        raise ValueError("I cannot extract momentum with that method....")
        
    print("CHECK THAT THIS IS NOT ZERO", check_norm)
    k_phase = np.angle(psi1/psi0) # well i am assuming that the phase depends only on k 
    k_phase = modulo_2pi(k_phase)
    k_int = round(L*k_phase/(2*np.pi),2)
    print("Phase ", k_phase)
    print("integer momentum",k_int  )
    return k_phase, k_int 

def Psi_polaron(w_window,K,N,U,U2,Vc,V0,alpha,g,mu,J,T,bias,L , tolerance =0.2,sps = None,M=100,Gamma=0.05, Ext_pot_q=0):
    """ Outputs the polaron wavefunction that is the brightest within some energy window and of a definite momentum 
    So the strategy is given some energy window and K to find the brightest couple (using the Lanczos code)
    Now that you have that couple of values so the right energy: That allows you to narrow down the 
    energy window then i will diagonalize the full hamiltonian and select the eigenstate within the right energy window 
    Then compute for all of the eigenstates their momenta and chose the one with the right momentum
    Input :
    w_window : energy window
    K : Polaron momentum
    Remark :  sigma is the target energy. Here I am computing the eigenvector and eigenvalue which is the closed to that w_bright
    I chose to compute 10 (should be L in general) eigenvalues that are close to that target because of the degeneracy  and I chose the eigenvector which 
    has the right momentum (this uses Lanczos  but maybe it is better to do it Brute-Force first)
    I never verify is there are states is the given energy window
    """
    if sps == None:
        sps = N+1
    n_range = np.arange(-(L//2), L//2 + (L % 2))
    qvals = 2*np.pi*n_range/L

    K_index, K_int = np.argmin(np.abs(qvals - K)),round(L*K/(2*np.pi),1)
    wmin, wmax = w_window
    w_range = np.linspace(wmin,wmax,200)


    Polaron_spec = Polaron_spectral_fct(L,N,U,U2,Vc,V0,alpha,g,mu,J, w_range , qvals,T,bias, k= None,sps = sps, M= M, Gamma = Gamma , Ext_pot_q= Ext_pot_q)
    w_bright_index = np.argmax(Polaron_spec[K_index,:])
    Z = np.max(Polaron_spec[K_index,:])*Gamma*np.pi
    w_bright = w_range[w_bright_index]
    print('OK found the brightest state with w bright = ', w_bright)
    print('with oscillator strength Z=',Z)
    H_bath, basis_bath = bath_Hamil(L,N,U,U2,Vc,V0,alpha,mu,J,T,k= None,sps = sps, Ext_pot_q=0)
    print('computed bath hamil and shape', H_bath.shape)
    Egs, _ =  H_bath.eigh()  
    print('bath GS E', Egs[0])
    full_basis,full_H =  polaron_Hamil(L,N,U,U2,Vc,V0,alpha,g,mu,J,T,bias,k= None,sps = sps)
    #H_sparse = full_H.tocsr() 
    #E_low, V_low = eigsh(H_sparse , k=L, sigma=w_bright, which='LM')
    E_low, V_low = full_H.eigh()
    E_low = E_low - Egs[0]
    w_min, w_max = w_window
    print('Energies are ', E_low)
    mask = (E_low >= w_bright -tolerance) & (E_low <= w_bright +tolerance)
    E_window = E_low[mask]
    V_window = V_low[:, mask]
    K_target = None
    index = 0
    print('len E window', len(E_window))
    while K_target == None and index< len(E_window):
        V = V_window[:,index]
        k,k_int  = polaron_momentum(V ,full_basis,L,N,M= M,Gamma =Gamma, Ext_pot_q=Ext_pot_q)
        index += 1
        if k_int == K_int :
            K_target = k 
            K_target_int = k_int  
    print('now index is', index)
    Psi_kw =  V_window[:,index-1]
    w = E_window[index-1]
    print('w is', w)

    return Psi_kw, w,k,k_int,full_basis,Z

