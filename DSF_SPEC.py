from Lanczos import *
L=24
a=1
N=3
n = N/L
J=1.0
mu=-2
U = 0
U= 2
U2 = 0 #nearest neighbour interaction (bad idea because needs half filling to have CDW)
sps = None #hardcore condition 
#sps= 3 
alpha = 3  #the power of the power law potential
Vc = 100/n**alpha # ext pot amplitude for the power law potential
Vc =20/n**alpha 
Vc =0
V0 = 80.2  #ext pot amplitude for the sine potential
V0 = 8.2
V0 = 0.5
#V0 = 0
Ext_pot_q = 2*np.pi/3
Ext_pot_q = np.pi/4
Ext_pot_phi = 0
g=-2
g=-5

Yukawa_lam = 1
Gamma=0.05
M = 59 # Lanczos iterations
M = 23 # Lanczos iterations
m_B = 1
# Translational symmetry
T = np.roll(np.arange(L), -1)
#allowed momenta
n_range = np.arange(-(L//2), L//2 + (L%2))
qvals = 2*np.pi*n_range/L

w =  np.linspace(-1, 10, 500)



#S_q_w = S_qw(L,N,U,U2,Vc,mu,J,Green_op_q,w,qvals,T,k=0,sps=None,M=M,Gamma=Gamma, Ext_pot_q=Ext_pot_q)

#uncomment if you want DSF and Polaron spectral response in the full Hilbert space
#S_q_w = S_qw(L,N,U,U2,Vc,V0,alpha,mu,J,Green_op_q,w,qvals,T,k=None,sps=sps,M=M,Gamma=Gamma, Ext_pot_q=Ext_pot_q)

start_t = time.time()
#A_q_w = Polaron_spectral_fct(L,N,U,U2,Vc,alpha,g,mu,J, w, qvals, T,k= None,sps = None, M= M, Gamma = 0.05, Ext_pot_q=Ext_pot_q)
A_q_w = Polaron_spectral_fct(L,N,U,U2,Vc,V0,alpha,g,mu,J, w, qvals, T,1,k= None,sps = sps, M= M, Gamma = 0.05, Ext_pot_q=Ext_pot_q)

print('Computed polaron spectral fct in ', time.time() - start_t,' seconds')
plot_dyn_structure_factor(L,N,U,U2,Vc,V0,alpha,g,M,A_q_w, qvals, w, Gamma, yaxis = r"$A(q,\Omega)$", obs = "Polaron Spectral Function", bound = 1,Ext_pot_q=Ext_pot_q,sps =sps ) 





start_t = time.time()
#uncomment this
Green_op_q = [[[ 'n',[[np.exp(-1j*q*i)/np.sqrt(L),i] for i in range(L)] ] ] for q in qvals]
S_q_w = S_qw(L,N,U,U2,Vc,V0,alpha,mu,J,Green_op_q,w,qvals,T,k=None,sps=sps,M=M,Gamma=Gamma, Ext_pot_q=Ext_pot_q)
print('Computed DSF in ', time.time() - start_t,' seconds')
print('S qw is ',np.shape(S_q_w) )
plot_dyn_structure_factor(L,N,U,U2,Vc,V0,alpha,g,M,S_q_w, qvals, w, Gamma,Ext_pot_q=Ext_pot_q,sps =sps)




