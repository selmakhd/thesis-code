from Lanczos import *
from embedding import *
from find_polaron_wavefunction import *
from matplotlib.colors import PowerNorm

L=12
N=2

n = N/L
U=0
U2 =0
Vc=0
alpha =3
V0=0
mu=-2
g=-0.5
sps = None
sps =2 

Gamma=0.05
M = 89 # Lanczos iterations
# Translational symmetry
T = np.roll(np.arange(L), -1)
#allowed momenta
n_range = np.arange(-(L//2), L//2 + (L % 2))
qvals = 2*np.pi*n_range/L

def lambda_Schmidt(w_window,K,N,U,U2,Vc,V0,alpha,g,mu,J,T,L = L, tolerance =0.2,sps = None,M=M):
    """  Outputs the Schmidt weight decomposition of a Polaron state
    input
    psi(w_window,K) : Polaron state in the total Hilbert space carrying momentum K 
    I know that psi = sum_{k} lambda(k,omega) |K-k> x |phi k>
    output
    lambda_table w rows and k columns I want to complete column by column. 
    """
    if sps == None:
        sps = N+1

    Psi_K,psiw,psik,k_int,full_basis,Z  = Psi_polaron(w_window,K,N,U,U2,Vc,V0,alpha,g,mu,J,T,1,L = L,tolerance = tolerance, sps =sps,M=M)
    Psi_K2,psiw2,psik2,k_int2,full_basis2,Z2  = Psi_polaron(w_window,K,N,U,U2,Vc,V0,alpha,g,mu,J,T,-1,L = L,tolerance = tolerance, sps =sps,M=M)
    
    basis_b = boson_basis_general(N=L, Nb=N, sps = sps)
    identity = [["I", [i], 1] for i in range(L)]
    lambda_table = [] #list of triplets (energy, momentum, lambda squared)
    lambda_table2 = []
    print('psik is', psik)
    psik = -psik

    for k in qvals: #instead of projecting or whatever Ah no i have to project or somehow embed
        kint = int(round(L*k/(2*np.pi)))
        H,basis_k = bath_Hamil(L,N,U,U2,Vc,V0,alpha,mu,J,T,k= kint,sps = sps, Ext_pot_q=0)
        E,V = H.eigh()
        Cq_imp_vac = np.exp(-1j*np.arange(L)*(psik-k)) / np.sqrt(L)
        Cq_imp_vac2 = np.exp(-1j*np.arange(L)*(psik2-k)) / np.sqrt(L)
        
        print('for k', k)
        for i,w in enumerate(E):
            if kint ==0 and i==0:
                Egs=w
                print('ground state energy is', Egs)
            phi_k = V[:,i]
            #phi_k= basis_b.project_to(basis_k.project_from(phi_k,sparse=False),sparse=False)
            phi_k=  embed_vector(phi_k, kint, basis_k, basis_b, L)
            phi_k /= np.linalg.norm(phi_k)
            
            schmidt_vec_k_w = np.kron(phi_k , Cq_imp_vac)
            schmidt_vec_k_w2 = np.kron(phi_k , Cq_imp_vac2)
            
            overlap = np.vdot(Psi_K, schmidt_vec_k_w)
            overlap2 = np.vdot(Psi_K2, schmidt_vec_k_w2)
            lambda_table.append((w,k, np.abs(overlap)**2))
            lambda_table2.append((w,k, np.abs(overlap2)**2))
    ##print('psiw - Egs is ', psiw-Egs)
    print('len lambda tables',len(lambda_table),len(lambda_table2), lambda_table[0] )
    #lambda_table_average = [(lambda_table[i] +lambda_table2[i])/2 for i in range(len(lambda_table))]
    lambda_table_average = [
    tuple(
        (a + b)/2
        for a, b in zip(lambda_table[i], lambda_table2[i])
    )
    for i in range(len(lambda_table))
]

    return lambda_table_average,psiw,psik,k_int,Z
    #return lambda_table,psiw,psik,k_int,Z




#lambda_table,w,psik,k_int,Z  = lambda_Schmidt((0,3.5),0,N,U,U2,Vc,V0,alpha,g,mu,J,T,L = L, tolerance=0.2,sps =sps,M=M)       #main branch
#lambda_table,w,psik,k_int,Z  = lambda_Schmidt((-0.2,1),-np.pi,N,U,U2,Vc,V0,alpha,g,mu,J,T,L = L)       #repulsive polaron
#lambda_table,w,psik,k_int,Z  = lambda_Schmidt((-0.2,1),-np.pi,N,U,U2,Vc,V0,alpha,g,mu,J,T,L = L, tolerance=0.2,sps =sps,M=M)       #repulsive polaron
##lambda_table,w,psik,k_int,Z  = lambda_Schmidt((0.5,1),-np.pi,N,U,U2,Vc,V0,alpha,g,mu,J,T,L = L, tolerance=0.2,sps =sps,M=M)       #repulsive polaron
lambda_table,w,psik,k_int,Z  = lambda_Schmidt((3,4),-np.pi,N,U,U2,Vc,V0,alpha,g,mu,J,T,L = L, tolerance=0.2,sps =2,M=M)       #repulsive polaron
#lambda_table,w,psik,k_int,Z  = lambda_Schmidt((-2,0),0,N,U,U2,Vc,V0,alpha,g,mu,J,T,L = L, tolerance=0.2,sps =sps,M=M)       #repulsive polaron

#lambda_table,w,psik,k_int,Z  = lambda_Schmidt((-5,0),0,N,U,U2,Vc,V0,alpha,g,mu,J,T,L = L)  #2-body bound state
#lambda_table,w,psik,k_int,Z  = lambda_Schmidt((-6,-5),0,N,U,U2,Vc,V0,alpha,g,mu,J,T,L = L)          #3-body bound state
#lambda_table,w,psik,k_int,Z  = lambda_Schmidt((-5,-2.5),0,N,U,U2,Vc,V0,alpha,g,mu,J,T,L = L, tolerance=0.2,sps =sps,M=M)          #3-body bound state
#lambda_table,w,psik,k_int,Z  = lambda_Schmidt((-2.5,-0.5),-np.pi,N,U,U2,Vc,V0,alpha,g,mu,J,T,L = L, tolerance=0.2,sps =sps,M=M)   #3-body bound state momentum -pi

def plot_lambda(lambda_table,w,psik,k_int , num_levels=50):
    """
    Plot lambda squared(omega,k) as a color map.
    lambda_table : list of (omega, k, lambda squared)
    """
    #print('plot fct')
    omega_vals, k_vals, lam2_vals = zip(*lambda_table)
    omega_vals = np.array(omega_vals)
    k_vals = np.array(k_vals)
    lam2_vals = np.array(lam2_vals)
    #total_prob = sum([triplet[2] for triplet in lambda_table])
    total_prob = sum(lam2_vals)
    #print('max of lambdas', np.max(lam2_vals))
    #print('list of lambdas', lam2_vals)
    print("Sum of lambda^2 over all bath states:", total_prob)

    plt.figure(figsize=(8, 6))
    #norm = mcolors.LogNorm(vmin=max(lam2_vals.min(), 1e-20), vmax=lam2_vals.max())
    norm = PowerNorm(gamma=0.5, vmin=lam2_vals.min(), vmax=lam2_vals.max())

    ##plt.tricontourf(k_vals, omega_vals, lam2_vals, levels=num_levels, cmap='inferno', norm = norm)
    #plt.tricontourf(k_vals, omega_vals, lam2_vals, levels=num_levels, cmap='inferno')
    
    ##sc = plt.scatter(k_vals, omega_vals, c=lam2_vals, cmap='inferno', s=250, edgecolors='grey',norm = norm)
    #plt.colorbar(label=r'$\lambda^2(k, \omega)$')
    ##plt.colorbar(sc, label=r'$\lambda^2(k, \omega)$')
    # scale dot sizes by lambda^2
    sizes = 3000* lam2_vals / lam2_vals.max()
    colors = lam2_vals
    sc = plt.scatter(
        k_vals,
        omega_vals,
        s=sizes,
        c=colors,
        cmap='viridis',       # any colormap you like
        norm=norm,  
        edgecolor='none',    # optional: outline for visibility
        linewidth=1.0,
        alpha=0.8             # slightly transparent to see overlapping points
        )

    plt.colorbar(sc, label=r'$\lambda^2(k, \omega)$')


    # scatter with size-coded dots (no color)
    #plt.scatter(
    #    k_vals,
    #    omega_vals,
    #    s=sizes,
    #    facecolor='white',
    #    edgecolor='black',
    #    linewidth=1.2
    #)

    # write lambda^2 value inside each dot
    #for x, y, val in zip(k_vals, omega_vals, lam2_vals):
    #    plt.text(
    #        x, y,
    #        f"{val:.2e}",    # notation like 1.23e-4
    #        ha='center', va='center',
    #        fontsize=8
    #    )

    
    down_curve, up_curve = [np.min(TG_excitations(q,N,L)) for q in k_vals], [np.max(TG_excitations(q,N,L)) for q in k_vals]
    plt.plot(k_vals, down_curve, c='r')
    plt.plot(k_vals, up_curve, c='r')
    plt.xlabel('k (bath momentum)')
    plt.ylabel(r'$\omega$ (bath energy)')
    plt.title(f"Schmidt weights for polaron K_int={k_int } and energy ={round(w,4)}, Z={round(Z,2)}, U={U}, g={g}, L={L}, N={N}, sps ={sps}")
    #plt.ylim(-1,10)
    plt.tight_layout()
    plt.show()

plot_lambda(lambda_table,w,psik,k_int )



def Entropy(w_window,K,N,U,U2,Vc,V0,alpha,g,mu,J,T,L = L):
    Psi_K,psiw,psik,k_int,full_basis,Z  = Psi_polaron(w_window,K,N,U,U2,Vc,V0,alpha,g,mu,J,T,L = L)
    #rdm_A = full_basis.partial_trace(Psi_K, sub_sys_A="left",return_rdm="A" , sparse=True)
    result = full_basis.ent_entropy( state=Psi_K, sub_sys_A="left",  return_rdm=None,  return_rdm_EVs=True )
    S_ent = result["Sent_A"]
    lambda2_vals = result["p_A"] #lam squared or proba
    return S_ent,  lambda2_vals


#print('entropy',Entropy((-1,1),0,N,U,U2,Vc,g,mu,J,T,L = L))
#print("log L", np.log(L))
gvals = np.linspace(0,5,10)
entropies = [Entropy((-0.1,0.5),0,N,U,U2,Vc,V0,alpha,g,mu,J,T,L = L) for g in gvals]
plt.plot(gvals,entropies)
plt.show()