from Lanczos import *

#L,N = 29,5
L,N = 23,4
U,g = 0, -5
M=49
# Translational symmetry
T = np.roll(np.arange(L), -1)
#allowed momenta
n_range = np.arange(-(L//2), L//2 + (L % 2))
qvals = 2*np.pi*n_range/L


w =  np.linspace(-10, 10, 300)

start_t = time.time()
A_q_w = Polaron_spectral_fct(L,N,U,g,mu,J, w, qvals,T, k= None, sps =2, M= M, Gamma = 0.05, Ext_pot_q=Ext_pot_q)
#print('Computed polaron spectral fct in ', time.time() - start_t,' seconds')
plot_dyn_structure_factor(L,N,U,g,M,A_q_w, qvals, w, Gamma, yaxis = r"$A(q,\Omega)$", obs = "Fermi Polaron Spectral Function", bound = 1,Ext_pot_q=Ext_pot_q ) 
