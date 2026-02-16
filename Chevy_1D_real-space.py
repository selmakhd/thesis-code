import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# So the structure is Bogoliubov - Dynamical structure factor - Spectral function - (correlation fct + impurity density)


a=1  #lattice spacing
flux =0         #If i want to try another gauge choice ! but here no minimum kept at  zero
n = 0.5
U= 0.1
L= 20
N = int(L*n)
g = -5
epsilon = 0.00000  #shift of the dynamical matrix for degeneracy lift
regularisation =0  
Gamma = 0.05       #broadening
mu = (-2 + U*n)
condensate = np.array([np.sqrt(n)*np.exp(+1j*flux*i) for i in range(L)])

def chem_potential(U,n):
    return -2 + U*n

def free_spec(k,mu):
    return -2*np.cos(k*a) -mu

def bogo_spec(k,U,n):
    Ek = free_spec(k-flux ,chem_potential(U,n))
    return np.sqrt(abs((Ek + U*n)*(Ek +3*U*n)))


def Aji(condensate,U):
    off =0.00000
    mu = -2 + U*n - regularisation
    mat = np.zeros((L,L),dtype = complex)
    for i in range(L):
        mat[i,(i+1)%L] = 1*np.exp(-1j*flux) + 1j*off
        mat[i,(i-1)%L] = 1*np.exp(1j*flux)  - 1j*off
        mat[i,i] = -2*U*abs(condensate[i])**2 + mu
    return mat

def Bji(condensate,U):
    mat = np.zeros((L,L),dtype = complex)
    for i in range(L):
        mat[i,i] = -U*condensate[i]**2
    return mat

def T_matrix_1D(pow,L):
    Tr = np.zeros((L, L), dtype=complex)
    zero = np.zeros_like(Tr)
    for j in range(L):
        Tr[j,(j+pow)%L] = 1
    return np.block([[np.exp(-1j*flux)*Tr, zero ],[zero ,np.exp(1j*flux)* Tr] ])

def eom(condensate, U,pow,L,epsilon = 0.00000000001):
    Translation = T_matrix_1D(pow,L)
    A = Aji(condensate,U)
    B = Bji(condensate,U)
    return np.block([[-A, -B ],[np.conj(B), np.conj(A) ] ]) +1*epsilon*(Translation  - np.conj(Translation).T )


#Raw Bogo spec
dyn_mat =eom(condensate, U,1,L)
eigenvalues_dyn_unsorted ,eigenvectors_dyn_unsorted= np.linalg.eig(dyn_mat)
sorted_indices = np.argsort(eigenvalues_dyn_unsorted.real)

#Sorted Bogo spec
eigenvalues_dyn, eigenvectors_dyn = eigenvalues_dyn_unsorted[sorted_indices].real, eigenvectors_dyn_unsorted[:, sorted_indices]

#Positive Bogo spec
Bogo_spec, Bogo_eigenvectors = eigenvalues_dyn_unsorted[sorted_indices][L+1:].real, eigenvectors_dyn_unsorted[:, sorted_indices][:,L+1:]

def bogo_plot(eigenvalues_dyn ,eigenvectors_dyn):
    k_vals = []
    spec = []
    spec_exact = []
    for i in range(len(eigenvalues_dyn)):
        k = (np.angle(eigenvectors_dyn[:,i][0]* np.conj(eigenvectors_dyn[:,i][1])) +flux) # -flux -q + flux
        k_vals.append(k)
        spec.append(eigenvalues_dyn[i])
        spec_exact.append(bogo_spec(k+flux ,U,n))
    plt.scatter(k_vals,spec)
    plt.scatter(k_vals,spec_exact, c ='r')
    plt.grid()
    plt.show()

#bogo_plot(eigenvalues_dyn ,eigenvectors_dyn)


def symplectic_normalization_1D(uv_eigenstate,L):  #eats L-1 columns and 2L rows 
    u_values = uv_eigenstate[:L,:]
    v_values = np.conj(uv_eigenstate[L:,:])
    symp_norm = np.sqrt( np.sum(np.abs(u_values)**2,axis = 0) - np.sum(np.abs(v_values)**2,axis =0) )
    new_u, new_v = u_values/symp_norm,v_values/symp_norm
    print('normalisation checkpoint', np.sqrt( np.sum(np.abs(new_u)**2,axis = 0) - np.sum(np.abs(new_v)**2,axis =0) ))
    return new_u, new_v 


def dyn_structure_factor_1D(q_vals, Omega_vals, Bogo_spec, condensate, Bogo_eigenvectors,L, Gamma=0.25):
    L = len(condensate)
    u_values, v_values = symplectic_normalization_1D(Bogo_eigenvectors,L)
    print("goldstone mode !!", u_values[:,0])
    phi_star_j = np.conj(condensate)[:, None, None, None]
    phi_j = condensate[:, None, None, None]
    vj_mu = v_values[:, :, None, None]  # index j, mu, q, omega
    u_star_j_mu = np.conj(u_values)[:, :, None, None]
    phase_factor = np.exp(1j * q_vals[None, None, :, None] * np.arange(L)[:, None, None, None])
    denominator = (Omega_vals[None, None, :] - Bogo_spec[:, None, None] + 1j * Gamma)
    super_oscillator_strength = phase_factor * (phi_star_j * vj_mu + phi_j * u_star_j_mu)
    oscillator_strength = np.abs(np.sum(super_oscillator_strength, axis=0))**2
    spec_matrix = np.sum(oscillator_strength / denominator, axis=0)
    return -2 / L**2 * spec_matrix.imag

def dyn_structure_factor_1D_plot(f, q_vals, Omega_vals, Bogo_spec, condensate, Bogo_eigenvectors,L, Gamma=0.25):
    spectral_values = f(q_vals, Omega_vals, Bogo_spec, condensate, Bogo_eigenvectors,L, Gamma)
    Q_mesh, Omega_mesh = np.meshgrid(q_vals, Omega_vals, indexing='ij')
    norm = mcolors.LogNorm(vmin=max(spectral_values.min(), 1e-4), vmax=spectral_values.max())
    
    plt.scatter(Q_mesh.flatten(), Omega_mesh.flatten(), c=spectral_values.flatten(), cmap='viridis', norm=norm, s=40)
    exact_spectrum = bogo_spec(q_vals, U, n)
    plt.plot(q_vals, exact_spectrum, color='red', linewidth=2, label=r'Exact Spectrum')  

    cbar = plt.colorbar(label=r"$S(q,\Omega)$")
    cbar.set_label(r"$S(q,\Omega)$", fontsize=15) 

    plt.xlabel(r"Momentum $q$", fontsize=15)
    plt.ylabel(r"Frequency $\Omega$", fontsize=15)
    plt.title(f"Dynamical Structure Factor L={L}, N={N}, U={U}," + r' $\Gamma=$' + f'{Gamma}', fontsize=15)
    plt.xticks(size=15)
    plt.yticks(size=15)
    #plt.xlim(-0.5,0.5)
    #plt.ylim(-0.5,0.5)
    plt.tight_layout()
    plt.show()

q_vals = 2*np.pi/L*np.arange(L) -np.pi 
Omega_vals = np.linspace(-10,10,2*L)

#dyn_structure_factor_1D_plot(dyn_structure_factor_1D, q_vals, Omega_vals, Bogo_spec, condensate, Bogo_eigenvectors,L, Gamma)


def WVrenorm(condensate, Bogo_eigenvectors,L):
    "m=1,2   j= 0,...,L -1    i = 0,..., 2L -1  "
    "Bogo_eigenvectors is a collection of columns (u v^*) it's a (2L) x(L-1) matrix"
    u, v = symplectic_normalization_1D(Bogo_eigenvectors,L) 
    W = np.conj(condensate)[:,None]* u + condensate[:,None]* np.conj(v)
    V = np.conj(u[:,None,:])*u[:,:,None] + np.conj(v[:,:,None])*v[:,None,:]
    renorm = abs(condensate)**2 + np.sum(np.abs(v)**2, axis=1)
    return W,V,renorm

W,V,renorm = WVrenorm(condensate, Bogo_eigenvectors,L)

def Chevy_mat(condensate,Bogo_spec,W,V,renorm, g):
    'I organise (l,i)  as (0, 0) (0,1) (0,2) ... (0,L-1) (1,0) (1,1) ... (1,L-1).... (L-1,0) ... (L-1,L-1)'
    ' ok so (j,index = i) = (whatever,0) then no bogo excitations  and (whatever, 1-->L) is with bogo excitation'
    matr = np.zeros((  L**2,  L**2 ),dtype = complex)
    for j in range(L):
        for j_prime in range(L):
            for index in range(L ):
                for index_prime in range(L ):
                    cste =  g*renorm[j] if (j==j_prime  and index ==index_prime) else 0 
                    bath_kin = Bogo_spec[(index-1)%L ] if (j==j_prime and index ==index_prime  and index !=0) else 0
                    impurity_kin = 0
                    if index == index_prime:
                        if (j + 1) % L == j_prime or (j - 1) % L == j_prime:
                            impurity_kin = -1
                        elif j == j_prime:
                            impurity_kin = 2 
                    W_int = 0
                    if index ==0 and index_prime !=0:
                        W_int =g* W[j,(index_prime-1)%L] if (j==j_prime) else 0
                    elif index_prime ==0 and index !=0:
                        W_int =g* np.conj(W[j,(index-1)%L]) if (j_prime == j) else 0
                    V_int =g *V[j,(index_prime-1)%L, (index-1)%L] if (j == j_prime and index !=0 and index_prime != 0) else 0
                    #bath_kin = 0
                    #impurity_kin =0
                    #V_int = 0
                    #W_int =0
                    #cste =0
                    matr[j*(L)+ index, j_prime*(L) +index_prime] = cste + bath_kin + impurity_kin + W_int + V_int
    return matr


chevy_matrix = Chevy_mat(condensate,Bogo_spec,W,V,renorm, g)

print('Hermiticity checkpoint', np.max(np.abs(chevy_matrix - np.conj(chevy_matrix).T)))
eigenvalues_chev, eigenvectors_chev = np.linalg.eigh(chevy_matrix)                          #spectrum of the chevy

def spec_func_1D_full(Q_vals, Omega_vals, eigenvalues_chev, eigenvectors_chev, L, Gamma):
    N_bogo = eigenvectors_chev.shape[1]
    j_vals = np.arange(L)
    phase_factors = np.exp(-1j * Q_vals[:, None] * j_vals[None, :]) / np.sqrt(L)
    psi_no_bogo = eigenvectors_chev[::L, :]  
    total_amplitudes = phase_factors @ psi_no_bogo 
    oscillator_strengths = np.abs(total_amplitudes) ** 2
    denom = Omega_vals[None, :, None] - eigenvalues_chev[None, None, :] + 1j*Gamma
    spectral_values = oscillator_strengths[:, None, :]/denom
    A = -2 * np.sum(spectral_values.imag, axis=2)
    return A


def spectral_function_plot_1D(Q_vals, Omega_vals, spectral_values, g, L, Gamma):
    Q_mesh, Omega_mesh = np.meshgrid(Q_vals, Omega_vals, indexing='ij')
    norm = mcolors.LogNorm(vmin=max(spectral_values.min(), 1e-6), vmax=spectral_values.max())
    plt.figure(figsize=(8, 6))
    plt.scatter(Q_mesh, Omega_mesh, c=spectral_values, cmap='viridis', norm=norm, s=40)
    plt.colorbar(label=r"$\mathcal{A}(\Omega,Q)$")
    plt.xlabel("Impurity momentum $Q$")
    plt.ylabel(r"$\Omega$")
    plt.title(fr"1D spectral function: $g = {g}, \Gamma = {Gamma}, L = {L}$")
    plt.tight_layout()
    plt.show()

Q_vals = 2*np.pi/L*np.arange(L) -np.pi 
Omega_vals = np.linspace(-10,10,10*L)
#spec_vals= spec_func_1D_full(Q_vals, Omega_vals, eigenvalues_chev, eigenvectors_chev, L, Gamma)
#spectral_function_plot_1D(Q_vals, Omega_vals, spec_vals, g, L, Gamma)



def compute_g2(i, chevy_vector, V, W, phi, L):
    g2 = 0
    psi = chevy_vector.reshape(L, L) 
    for j in range(L):
        phi_sq = np.abs(phi[(j + i) % L])**2
        g2 += phi_sq * np.abs(psi[j, 0])**2
        for nu in range(L - 1):
            g2 += phi_sq * np.abs(psi[j, nu + 1])**2
            g2 += W[(j + i) % L, nu] * np.conj(psi[j, 0]) * psi[j, nu + 1]
            g2 += np.conj(W[(j + i) % L, nu]) * np.conj(psi[j, nu + 1]) * psi[j, 0]
            for mu in range(L - 1):
                g2 += V[(j + i) % L, nu, mu] * np.conj(psi[j, mu + 1]) * psi[j,nu+1]
    g2 /= L
    return g2.real


def brightest_chevy_state_1D(Q, Omega_min, Omega_max, eigenvalues_chev, eigenvectors_chev, W, V, phi, L, Gamma=0.25, plot=True):
    N_chev = eigenvalues_chev.shape[0]
    j_vals = np.arange(L) 
    phase_factors = np.exp(-1j*Q*j_vals) / np.sqrt(L)
    psi_j0 = eigenvectors_chev[np.arange(L)*L + 0, :] 
    #print('psij0',psi_j0)
    total_amplitudes = phase_factors @ psi_j0
    #print('tot ampl',total_amplitudes)
    Z_n = np.abs(total_amplitudes)**2   #for a given Q

    energy_mask = (eigenvalues_chev >= Omega_min) & (eigenvalues_chev <= Omega_max)
    #print('energy mask', energy_mask)
    #print('Zn energy mask', Z_n)
    if not np.any(energy_mask):
        print(f"no states in energy window [{Omega_min}, {Omega_max}]")
        return None
    
    brightest_index = np.where(energy_mask)[0][np.argmax(Z_n[energy_mask])]
    psi_max = eigenvectors_chev[:, brightest_index]
    E =eigenvalues_chev[brightest_index]
    Z = Z_n[brightest_index]

    psi = psi_max.reshape(L, L)
    rho = np.sum(np.abs(psi)**2, axis=1)

    g2 = np.zeros(L)
    for i in range(L):
        g2[i] = compute_g2(i, psi_max, V, W, phi, L)

    if plot:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(np.arange(L), rho, marker='o', color='red')
        plt.xlabel("Site j",size=20)
        plt.ylim(0,1/L*3)
        plt.ylabel(r"$\rho_{imp}$",size =20)
        plt.xticks(fontsize =15)
        plt.yticks(fontsize =15)
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(np.arange(L), g2.real, marker='x', color='green')
        plt.xlabel("Site j",size = 20)
        plt.ylabel(r"$g^{(2)}$",size =20)
        plt.xticks(fontsize =15)
        plt.yticks(fontsize =15)
        plt.grid(True)

        plt.suptitle( fr"Brightest Chevy state : Q={Q:.2f}, $\Omega \in$[{Omega_min:.2f}, {Omega_max:.2f}], " f"E={E:.4f}, Z={Z:.4f}, U={U}, g={g}, L={L}, N={N}" ,size =20)
        plt.tight_layout()
        plt.show()


result = brightest_chevy_state_1D(  Q=0, Omega_min=-8, Omega_max=-5, eigenvalues_chev=eigenvalues_chev, eigenvectors_chev=eigenvectors_chev,   W=W, V=V,   phi=renorm,L=L) 



