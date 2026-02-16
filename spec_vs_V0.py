from Lanczos import *

L=24
a=1
N=3
n = N/L
J=1.0
mu=-2
U=0
U2=0
alpha = 3  #the power of the power law potential
Vc = 0/n**alpha #ext pot amplitude for the power law potential
V0 = 0  #ext pot amplitude for the sine potential
g=-5
Ext_pot_q = 2*np.pi/3
Ext_pot_q = -np.pi/4
#Ext_pot_phi = 0
sps = None
Yukawa_lam = 1
Gamma=0.05
M = 23 # Lanczos iterations
M = 49
m_B = 1
# Translational symmetry
T = np.roll(np.arange(L), -1)
#allowed momenta
n_range = np.arange(-(L//2), L//2 + (L%2))
qvals = 2*np.pi*n_range/L
w =  np.linspace(-18, 18, 500)
Green_op_q = [[[ 'n',[[np.exp(-1j*q*i)/np.sqrt(L),i] for i in range(L)] ] ] for q in qvals]



def spec_Vc( q,L,N,g,U,U2,Vc,alpha,mu,J, Green_op_q, wvals, V0vals,T, k =None,sps =None, M= 100, Gamma = 0.05, Ext_pot_q=0):
    'cut momentum q in the spec fct. k sector restriction ? no'
    if sps is None:
        sps = N+1
    q_index = int(round(L*q/(2*np.pi))) + L//2
    print('just a check q index is', q_index)
    print("q equals qvals[i] ???",q, ' and ', qvals[q_index])

    table = np.zeros( (len(V0vals), len(wvals)), dtype = float)
    for i,Vc in enumerate(V0vals):
        print('Vc index is',i)
        print('bath arguments', L,N,U,U2,Vc,V0,alpha,mu,J,T,k,sps , Ext_pot_q )
        bath_H, bath_basis = bath_Hamil(L,N,U,U2,Vc,V0,alpha,mu,J,T,k = k,sps = sps, Ext_pot_q=Ext_pot_q) 
        print('polaron H arguments', L,N,U,U2,Vc,V0,alpha,g,mu,J,T,1,k ,sps )
        full_basis, full_H = polaron_Hamil(L,N,U,U2,Vc,V0,alpha,g,mu,J,T,1,k = k,sps =sps)
        #print('here ok Vc,', Vc,L,N,U,U2,Vc,V0,g,mu,J,wvals)
        #print('alpha is ', alpha)
        print("about to compute polaron spec for given q and Vc", q,Vc)
        print('the arguments are ', L,N,U,U2,Vc,V0,g,mu,J, q, k , sps,M, Gamma )
        table[i,:] = full_procedure_polaron(bath_H,full_H,bath_basis,L,N,U,U2,Vc,V0,g,mu,J,wvals,q=q,k =k,sps = sps, M= M, Gamma = Gamma)
    print('table done')
    return table


V0vals = [i/10 for i in range(25)]
time_s = time.time()
spec_vs_Vc = spec_Vc( 0.0,L,N,g,U,U2,Vc,alpha,mu,J, Green_op_q, w, V0vals,T, k =None,sps =sps, M= M, Gamma = 0.05, Ext_pot_q=Ext_pot_q)
#spec_vs_Vc = spec_Vc( -np.pi,L,N,g,U,U2,Vc,alpha,mu,J, Green_op_q, w, V0vals,T, k =None,sps =sps, M= M, Gamma = 0.05, Ext_pot_q=Ext_pot_q)

print("this took", time_s - time.time(), "secondes...")
print('Vc is ', Vc)
plot_dyn_structure_factor(L,N,U,U2,Vc,V0,alpha,g,M,spec_vs_Vc, V0vals, w, Gamma, yaxis = r"$A(q,\Omega)$", obs = "Polaron Spectral Function", bound = 1,Ext_pot_q=Ext_pot_q,sps =sps, xaxis = 'Sine V0' ) 
