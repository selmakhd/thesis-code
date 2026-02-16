from Lanczos import *

L=14
a=1
N=3
n = N/L
J=1.0
mu=-2
U=0
U2=0
Vc = 0 # ext pot amplitude for the power law potential
alpha = 3  #the power of the power law potential
V0 = 0  #ext pot amplitude for the sine potential
g=-5
Ext_pot_q = 2*np.pi/3
Ext_pot_phi = 0
sps = None
Yukawa_lam = 1
Gamma=0.05
M = 45 # Lanczos iterations
m_B = 1
# Translational symmetry
T = np.roll(np.arange(L), -1)
#allowed momenta
n_range = np.arange(-(L//2), L//2 + (L%2))
qvals = 2*np.pi*n_range/L
w =  np.linspace(-20, 30, 500)
Green_op_q = [[[ 'n',[[np.exp(-1j*q*i)/np.sqrt(L),i] for i in range(L)] ] ] for q in qvals]


def spec_g( q,L,N,U,U2,Vc,V0,alpha,mu,J, Green_op_q, wvals, gvals,T, k =None,sps =None, M= 100, Gamma = 0.05, Ext_pot_q=0):
    " A(q=0, Omega) for varying g"
    if sps is None:
        sps = N + 1
    q_index = int(round(L*q/(2*np.pi))) + L//2 #assumes q is between -pi and pi
    print('just a check q index is', q_index)
    print("q equals qvals[i] ???",q, ' and ', qvals[q_index])
    Green_operator = Green_op_q[q_index]
    #bath_H, bath_basis = bath_Hamil(L,N,U,U2,Vc,alpha,mu,J,T,k = k,sps = sps, Ext_pot_q=Ext_pot_q) 

    table = np.zeros( (len(gvals), len(wvals)), dtype = float)
    for i,g in enumerate(gvals):
        print('g index is',i)
        bath_H, bath_basis = bath_Hamil(L,N,U,U2,Vc,V0,alpha,mu,J,T,k = k,sps = sps, Ext_pot_q=Ext_pot_q) 
        full_basis, full_H = polaron_Hamil(L,N,U,U2, Vc,V0,alpha,g,mu,J,T,1,k = k,sps =sps)
        
        #full_basis, full_H = polaron_Hamil(L,N,U,U2, Vc,alpha,g,mu,J,T,k = k,sps =sps)
        #table[i,:] = full_procedure_polaron(bath_H,full_H,bath_basis,L,N,U,U2,Vc,g,mu,J,wvals,q=q,k =k,sps = sps, M= M, Gamma = Gamma)
    
        table[i,:] = full_procedure_polaron(bath_H,full_H,bath_basis,L,N,g,U2,Vc,V0,U,mu,J,wvals,q=q,k =k,sps = sps, M= M, Gamma = Gamma)
    return table


gvals = np.linspace(-0.5,0.5,15)
gvals = np.linspace(-20.5,20.5,25)
#Uvals = np.linspace(0,5,15)
spec_vs_g = spec_g(0,L,N,U,U2,Vc,V0,alpha,mu,J, Green_op_q, w, gvals,T, k =None,sps =sps, M= 100, Gamma = 0.05, Ext_pot_q=0)
#spec_vs_g = spec_g( 0,L,N,U,U2,Vc,V0,alpha,mu,J, Green_op_q, w, gvals,T, k =None,sps =sps, M= 100, Gamma = 0.05, Ext_pot_q=0)
print('shape of the table', spec_vs_g.shape)
print('np shape', np.shape(spec_vs_g))

plot_dyn_structure_factor(L,N,U,U2,Vc,V0,alpha,g,M,spec_vs_g, gvals, w, Gamma, yaxis = r"$A(q,\Omega)$", obs = "Polaron Spectral Function", bound = 1,Ext_pot_q=Ext_pot_q,sps =sps, xaxis = 'g' ) 


