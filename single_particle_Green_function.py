from Lanczos import *
import numpy as np
import matplotlib.pyplot as plt
import time

L = 12
a = 1
N = 2
n = N / L
J = 1.0
mu = -2
U = 2
U2 = 0  # nearest neighbour interaction
sps = N+1
Vc = 0
alpha = 3
g = 0
V0 = 0
Ext_pot_q = 2 * np.pi / 3
Ext_pot_phi = 0
Yukawa_lam = 1
Gamma = 0.25
M = 29
m_B = 1

# Translational symmetry
T = np.roll(np.arange(L), -1)

# allowed momenta
n_range = np.arange(-(L // 2), L // 2 + (L % 2))
qvals = 2 * np.pi * n_range / L
w = np.linspace(-10, 10, 500)


def single_particle_Green(L, N, U, mu, J, q_vals, Omega_vals, Gamma=0.25):
    # Ground-state sector (N particles)
    HN, basisN = bath_Hamil(L, N, U, U2, Vc, mu, J, T, sps=sps)
    EN, VN = HN.eigh()
    E0 = EN[0]
    psi0 = VN[:, 0]
    psi0 = basisN.project_from(psi0, sparse=False, pcon=False)
    print("Ground state energy:", E0)

    #N+1 and N-1 sectors
    Hp, basisP = bath_Hamil(L, N + 1, U, U2, Vc, mu, J, T, sps=sps)
    Ep, Vp = Hp.eigh()

    Hm, basisM = bath_Hamil(L, N - 1, U, U2, Vc, mu, J, T, sps=sps)
    Em, Vm = Hm.eigh()

    full_basis =  boson_basis_general(N=L, sps = sps)

    num_q = len(q_vals)
    num_omega = len(Omega_vals)
    G_kw = np.zeros((num_q, num_omega), dtype=np.complex128)

    for iq, q in enumerate(q_vals):
        print("done with q =", q)

        adag_q_list = [['+', [[np.exp(-1j * q * j) / np.sqrt(L), j] for j in range(L)]]]
        adag_q = hamiltonian(adag_q_list, [], basis=full_basis, dtype=np.complex128, check_herm=False, check_pcon=False)

        vec_p = adag_q.dot(psi0)
        vec_p =  basisP.project_to(vec_p, sparse=False, pcon=False)
    

        osc_p = np.abs(Vp.conj().T @ vec_p) ** 2

        a_q_list = [['-', [[np.exp(1j * q * j) / np.sqrt(L), j] for j in range(L)]]]
        a_q = hamiltonian(a_q_list, [], basis=full_basis, dtype=np.complex128, check_herm=False, check_pcon=False)

        vec_m = a_q.dot(psi0)
        vec_m = basisM.project_to(vec_m, sparse=False, pcon=False)
        osc_m = np.abs(Vm.conj().T @ vec_m) ** 2

        for iomega, omega in enumerate(Omega_vals):
            denom_p = omega + E0 - Ep + 1j * Gamma
            denom_m = omega - E0 + Em + 1j * Gamma
            G_kw[iq, iomega] = np.sum(osc_p / denom_p) + np.sum(osc_m / denom_m)

    return G_kw


start_t = time.time()

G_kw = single_particle_Green(L, N, U, mu, J, qvals, w, Gamma)
A_kw = -1/np.pi*G_kw.imag

print('Computed single-particle spectral function in ', time.time() - start_t, ' seconds')



def plot_single_Green(L,N,U,U2,Vc,alpha,g,M,S_qw, q_vals, Omega_vals, Gamma=0.05, yaxis = r"$S(q,\Omega)$", obs = "Dynamical Structure Factor",Ext_pot_q=0, bound = None,sps = None):
    print('heres, sqw shape', np.shape(S_qw))
    Q_mesh, Omega_mesh = np.meshgrid(q_vals, Omega_vals, indexing='ij')
    norm = mcolors.LogNorm(vmin=max(S_qw.min(), 1e-6), vmax=S_qw.max())
    #norm = mcolors.Normalize(vmin=S_qw.min(), vmax=S_qw.max())
    
    plt.figure(figsize=(8,6))

    plt.pcolormesh(Q_mesh, Omega_mesh, S_qw, cmap='viridis', norm=norm, shading='auto')
    
    cbar = plt.colorbar(label=yaxis)
    cbar.set_label(yaxis, fontsize=14)
    x = np.linspace(-np.pi,np.pi,300)
    

    plt.xlabel("Momentum q", fontsize=14)
    plt.ylabel(r"Frequency $\Omega$", fontsize=14)
    plt.title(obs+rf", M={M+1}, $\Gamma$={Gamma}, U={U}, U2={U2}, Vc={round(Vc,3)}, $\alpha=${alpha}, g={g}, L={L}, N={N}, sps={sps}", fontsize=14)

    plt.tight_layout()
    plt.show()

plot_single_Green(L,N,U,U2,Vc,alpha,g,M,A_kw, qvals, w, Gamma=0.05, yaxis = r"$S(q,\Omega)$", obs = "Spectral function",Ext_pot_q=0, bound = None,sps = None)
    


plt.imshow(
    np.log(A_kw.T),
    origin='lower',
    aspect='auto',
    extent=[qvals[0], qvals[-1], w[0], w[-1]]
)


plt.xlabel(r"$k$")
plt.ylabel(r"$\omega$")
plt.colorbar(label=r"$A(k,\omega)$")
plt.show()
