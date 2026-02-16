from Lanczos import *
from scipy.stats import binom


L,N = 23,4
U,g,U2,Vc = 0, -5, 0,0

M=49
w =  np.linspace(-20, 10, 300)
# Translational symmetry
T = np.roll(np.arange(L), -1)
#allowed momenta
n_range = np.arange(-(L//2), L//2 + (L % 2))
qvals = 2*np.pi*n_range/L
p=1/L
#prob = [binom.pmf(k, N, p) for k in np.arange(N+1) ]


def Polaron_spec_vs_energy(q,L,N,U,U2,Vc,g,mu,J, wvals, qvals, k= None,sps = None, M= 100, Gamma = 0.05, Ext_pot_q=Ext_pot_q):
    if sps is None:
        sps = N+1
    bath_H, bath_basis = bath_Hamil(L,N,U,U2,Vc,mu,J,T,k= k,sps = sps, Ext_pot_q=Ext_pot_q) 
    full_basis, full_H = polaron_Hamil(L,N,U,U2,Vc,g,mu,J,T,k = k,sps =sps)
    A = np.zeros(  len(wvals), dtype = float)
    #i = qvals.index(q)
    qint = int(round(L*q/(2*np.pi)))
    A[:] = full_procedure_polaron(bath_H,full_H,bath_basis,L,N,U,U2,g,Vc,mu,J,wvals,q=q,k =k,sps = sps, M= M, Gamma = Gamma)
   
    return A
#A = Polaron_spec_vs_energy(0,L,N,U,U2,Vc,g,mu,J, w, qvals, k= None,sps = N+1, M= M, Gamma = 0.05, Ext_pot_q=Ext_pot_q)
N_list = [[2,16],[3,24],[4,32],[5,40]]
N_list = [[5,29]]
N_list = [[2,16],[3,24],[4,32]]
#plt.figure(figsize=(7,4))
for e in N_list:
    N,L = e
    #A = Polaron_spec_vs_energy(0,L,N,U,U2,Vc,g,mu,J, w, qvals, k= None,sps = N+1, M= M, Gamma = 0.05, Ext_pot_q=Ext_pot_q)
    A = Polaron_spec_vs_energy(0,L,N,U,U2,Vc,g,mu,J, w, qvals, k= None,sps = None, M= M, Gamma = 0.05, Ext_pot_q=Ext_pot_q)
    
    plt.plot(w, np.log(A), lw=2, label =  fr"Polaron spectral function  (N={N},  density={N/L:.3f})")
    #plt.plot(w, A, lw=2, label =  fr"Polaron spectral function  (N={N},  density={N/L:.3f})")
    
    #plt.plot(w, A, lw=2, label =  fr"Polaron spectral function  N={N}, L={L},  density={N/L:.3f}")

plt.plot([-2.3,-5.2,-8,-12],2.5+np.log([binom.pmf(1, N, p),binom.pmf(2, N, p),binom.pmf(3, N, p),binom.pmf(4, N, p)]))
plt.xlabel(r"Energy $\omega$", fontsize=14)
plt.ylabel(r"Spectral weight $A(k\!=\!0,\omega)$", fontsize=14)

plt.title(fr"U={U}, U2={U2}, Vc={Vc},  g={g},M={M}, $\Gamma$={Gamma}", fontsize=15)
plt.legend(fontsize = 15)
plt.grid()
#plt.tight_layout()
plt.show()

