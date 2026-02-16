from Chevy_1D_polaron_spec import *

a=1
L=30
n = 0.05
n= 2/L # for second order Chevy
N = n*L
U=0.5
U=0
g = -2
Gamma = 0.05 # broadening
multiple = 1 
cutoff = multiple*2*np.pi # for 3D
#m_I = 1/2
#_B = 1/2  #we assume mass of bosons = mass of impurity
#dk = 2*np.pi/L
SSB_lambda = 0


def Chevy_analysis_2Boglons(p,w_window,U,n,g,L,SSB_lambda):
    """ I take the eigenstates of the Hamiltonian such that
    their eigenvalue is inside the omega window and I chose 
    the one with the highest overlap with the pertubation 
    operator I chose.
    w_window is a couple (min,max)
    """
    w_min, w_max = w_window
    
    Ham = Hamiltonian_2Boglons_approx(p, U, n, g, L, SSB_lambda)
    E, V = np.linalg.eigh(Ham)
    mask = (E >= w_min) & (E <= w_max)

    if not np.any(mask):
        print("No eigenvalues in the specified window.")
        return None, None, None

    E_window = E[mask]
    V_window = V[:, mask]

    overlaps = np.abs(V_window[0,:])**2
    best_index = np.argmax(overlaps)
    
    best_state = V_window[:, best_index]
    best_energy = E_window[best_index]
    overlap = overlaps[best_index]
    
    return best_energy, best_state, overlap
def Chevy_analysis_2Boglons_plot(p,w_window,U,n,g,L,SSB_lambda):
    best_energy, best_state, overlap = Chevy_analysis_2Boglons(p,w_window,U,n,g,L,SSB_lambda)
    k1k2 = [(k1,k2 ) for k1 in range(L) for k2 in range(k1+1)] 

    best_state = np.abs(best_state)

    psi_k = np.array([best_state[i] for i, (k1,k2) in enumerate(k1k2) if k2 == 0 and k1 != 0]) 

    psi_kk = np.zeros((L,L), dtype=float)
    for i, (k1,k2) in enumerate(k1k2):
        #convert back k1 k2 into range 0 2pi
        k1,k2 = k1%L, k2%L
        if k2 != 0:
            psi_kk[k1,k2] = best_state[i]
            psi_kk[k2,k1] = best_state[i]
    
    psi_k = np.roll(psi_k,-L//2)
    psi_kk = np.roll(psi_kk,-L//2, axis=(0,1))

    qvals = 2*np.pi/L*np.arange(L) -np.pi
    nonzeros = psi_kk[psi_kk != 0]
    smallest_nonzero = nonzeros.min() if nonzeros.size > 0 else None

    fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(10,4))
    ax1.plot(qvals[1:], psi_k)
    ax1.set_xlabel('momentum k')
    ax1.set_ylabel(r"$|\Psi_k|$")
    ax1.set_title(rf" U={U}, n={round(n,3)}, L={L}, $\Gamma$={Gamma}, $\lambda=${SSB_lambda}")

    k1_mesh, k2_mesh = np.meshgrid(qvals,qvals, indexing ='ij')
    norm = mcolors.LogNorm(vmin= smallest_nonzero, vmax=psi_kk.max())

    pcm = ax2.pcolormesh(k1_mesh, k2_mesh, psi_kk, cmap='viridis', norm=norm, shading='auto')
    cbar = fig.colorbar(pcm, ax=ax2, label=r"$|\Psi(k_1,k_2)|$")
    cbar.set_label(r"$|\Psi(k_1,k_2)|$", fontsize=14)

    ax2.set_title(rf"p={round(p,3)}, E={round(best_energy,3)}, Z={round(overlap,3)}")
    ax2.set_xlabel(r'$k_1$')
    ax2.set_ylabel(r'$k_2$')
 
    plt.tight_layout()
    plt.show()
    return None


def spectral_function_plot_table(Q_vals, Omega_vals, spectral_vals, g, L, U, n, eps):
    Q_mesh, Omega_mesh = np.meshgrid(Q_vals, Omega_vals, indexing='ij')

    norm = mcolors.LogNorm(vmin=max(spectral_vals.min(), 1e-6), vmax=spectral_vals.max())

    plt.figure(figsize=(8, 6))

    #extent = [Q_vals.min(), Q_vals.max(), Omega_vals.min(), Omega_vals.max()]
    #im = plt.imshow(spectral_vals, extent=extent, origin='lower',
    #                aspect='auto', cmap='viridis', norm=norm, interpolation='bicubic')

    #cbar = plt.colorbar(im, label=r"$\mathcal{A}(\Omega,Q)$")
    #cbar.set_label(r"$\mathcal{A}(\Omega,Q)$", fontsize=15) 
    
    plt.pcolormesh(Q_mesh, Omega_mesh, spectral_vals.T, cmap='viridis', norm=norm, shading='auto')

    cbar = plt.colorbar(label=r"$\mathcal{A}(\Omega,Q)$")
    cbar.set_label(r"$\mathcal{A}(\Omega,Q)$", fontsize=14)
    
    plt.xlabel("Impurity momentum $Q$", size=15)
    plt.ylabel(r"$\Omega$", size=15)
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.title(rf"$g={g}$, $\Gamma={Gamma}$, $L={L}$, $U={U}$, $n={n}$", size = 15)
    plt.tight_layout()
    plt.show()


def compute_rho_0_spectral_function(Q_vals, Omega_vals,U,n, gamma,L,SSB_lambda):
    k_vals = np.arange(L)*(2*np.pi/L)  
    q_grid, k_grid = np.meshgrid(Q_vals, k_vals, indexing='ij')  
    shifted_k = (q_grid - k_grid)  
    omega_I = spec_impurity(shifted_k) 
    omega_B = bogo_spec(k_vals, U, n,SSB_lambda)[None, :]
    total_energy = omega_B + omega_I + g*n + g/L*sum( vk_eta(2*np.pi/L*k_1,U,n,SSB_lambda)**2  for k_1 in range(1,multiple*L))
    max_energy_Q = np.max(total_energy, axis=1) 
    min_energy_Q = np.min(total_energy, axis=1)  
    Omega_vals_expanded = Omega_vals[:, None, None]  
    energy_diff = Omega_vals_expanded - total_energy[None, :, :] 
    lorentzian = gamma / (np.pi * (energy_diff ** 2 + gamma ** 2))
    spectral_values = np.sum(lorentzian, axis=2) 
    return spectral_values, max_energy_Q, min_energy_Q
def plot_rho_0_spectral_function(Q_vals, Omega_vals, spectral_vals):
    Q_mesh, Omega_mesh = np.meshgrid(Q_vals, Omega_vals, indexing='ij')
    norm = mcolors.LogNorm(vmin=max(spectral_vals.min(), 1e-6), vmax=spectral_vals.max())

    plt.figure(figsize=(8, 6))
    plt.scatter(Q_mesh, Omega_mesh, c=spectral_vals.T, cmap='viridis', norm=norm, s=40)
    plt.colorbar(label=r"$\rho_0(\Omega,Q)$")  
    plt.xlabel("Impurity momentum $Q$", fontsize=14)
    plt.ylabel(r"$\Omega$", fontsize=14)
    plt.title(r" $\rho_0(\Omega,Q)$ for " + f'U={U}, n={n}, L={L}', fontsize=16)
    plt.tight_layout()
    plt.show()


def GS_vs_lambda(lambda_list, U_list):
    """Outputs the Energy and brightness of the brightest state at q =0 withing omega  in [omega_min, omega_max] VS lambda
    """   
    for U in U_list:
        GS_energy = []
        for lam in lambda_list:
            Boson_GS_energy = bogo_spec(0,U,n,-lam)*n*L
            Boson_GS_energy = 0
            print("lambda is ", lam)
            # Build Hamiltonian at Q=0
            H =  Frolich_Hamiltonian_new(0, U, n, g, L, -lam )
            eigvals, eigvecs = np.linalg.eigh(H)
            # Ground state energy
            gs_E = eigvals[0] - Boson_GS_energy 
            GS_energy.append(gs_E)
        plt.plot(lambda_list, GS_energy , label = f'U = {round(U,3)}')
    plt.title(rf'Ground state polaron energy VS $\lambda$ L={L}, n={n}, g={g}  ', size=30 )
    plt.xlabel(r"$\lambda$", size = 30)
    plt.ylabel("Ground state energy", size = 30)
    plt.legend(fontsize = 30)
    plt.grid()
    plt.show()
    #s= spectral_function_table([0], Omega_range, U, n, g, L, Gamma, Frolich_Hamiltonian_new,SSB_lambda)
    return None


##Chevy_analysis_2Boglons_plot(0,(-7,-5),U,n,g,L,SSB_lambda)

##lambda_list = np.linspace(0,0.001,20)
##U_list = np.linspace(0.05,0.15,4)
#GS_vs_lambda(lambda_list, U_list)


Q_vals = np.array([2*np.pi/L*i  -np.pi for i in range(multiple*L) ])
Omega_vals = np.linspace(-12,10,10*L)
Omega_vals = np.linspace(-2.5,8,10*L)
#spec_vals = spectral_function_table(Q_vals, Omega_vals, U, n, g, L, Gamma, Frolich_Hamiltonian_new,SSB_lambda)
##spec_vals = spectral_function_table(Q_vals, Omega_vals, U, n, g, L, Gamma, Hamiltonian_2Boglons_approx,SSB_lambda)

#spectral_vals, max_energy_Q, min_energy_Q = compute_rho_0_spectral_function(Q_vals, Omega_vals, U, n, Gamma, L,SSB_lambda)
#plot_rho_0_spectral_function(Q_vals, Omega_vals, spectral_vals)
##spectral_function_plot_table(Q_vals, Omega_vals, spec_vals, g, L, U, n, Gamma)

