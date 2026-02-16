from matplotlib.colors import PowerNorm
from Chevy_1D_polaron_spec import *
from Chevy_1D_momentum import *

def schmidt_table(p, w_window, U, n, g, L, SSB_lambda, num_levels=50):
    """
    Compute Schmidt triplets (omega_B, k_B, lambda_B) for the polaron
    state in a given energy window and plot using interpolation.

    Remark: qval is between -pi and pi but the k1k2 pairs are integers where 0 correspond to momentum -pi
    """
    best_energy, best_state, overlap = Chevy_analysis_2Boglons(p, w_window, U, n, g, L, SSB_lambda)
    if best_state is None:
        print("No state found in the given window.")
        return None

    #qvals = 2*np.pi/L * np.arange(L) -np.pi  #OK if L is even
    qvals = 2*np.pi/L * np.arange(L)  
    k1k2 = [(k1, k2) for k1 in range(L) for k2 in range( k1+1)] #k2 always smaller

    triplets = []
    triplets.append([0.0, 0.0, best_state[0]])  # zero-bogolon can only contribute with energy 0 I guess or maybe it is better to not include it since it is not the bath..
    #triplets.append([0.0, 0.0, 0])
    for k in range(1, L):
        omeg_B = bogo_spec(qvals[k], U, n, SSB_lambda)
        kB = qvals[k]
        amp_idx = [i for i, (k1, k2) in enumerate(k1k2) if k1 == k and k2 == 0]
        print('len amp idx',len(amp_idx))
        triplets.append([omeg_B, kB, best_state[amp_idx[0]]])

    for i, (k1, k2) in enumerate(k1k2):
        if k2 == 0: 
            continue
        amp = best_state[i]
        omeg_B = bogo_spec(qvals[k1], U, n, SSB_lambda) + bogo_spec(qvals[k2], U, n, SSB_lambda)
        #kB = (qvals[k1] + qvals[k2] + np.pi) % (2*np.pi) - np.pi
        kB = (qvals[k1] + qvals[k2]) % (2*np.pi)
        triplets.append([omeg_B, kB, amp])

    triplets = np.array(triplets, dtype=object)
 
    lam_sq = np.array([abs(c[2])**2 for c in triplets])
    omega_vals = np.array([c[0] for c in triplets], dtype=float)

    k_vals = np.array([c[1] for c in triplets], dtype=float)

    print(f"Total # of configurations: {len(triplets)}")
    print(f"Sum |lam|² = {lam_sq.sum():.6f}")
    #  shift the plot by half 
    for i in range(len(k_vals)):
        if k_vals[i]>= np.pi:
            k_vals[i] = k_vals[i] - 2*np.pi

    plt.figure(figsize=(8,6))
    sizes = 500 * lam_sq / lam_sq.max()   # normalize sizes

    norm = PowerNorm(gamma=0.5, vmin=lam_sq.min(), vmax=lam_sq.max())
    sc = plt.scatter(
        k_vals, omega_vals,
        s=sizes,          
        c=lam_sq,         
        cmap='viridis',
        norm=norm,
        edgecolors='None',
        alpha=0.9
    )

    #plt.tricontourf(k_vals, omega_vals, lam_sq, levels=num_levels, cmap='inferno', norm=norm)
    #sc = plt.scatter(k_vals, omega_vals, c=lam_sq, cmap='inferno', s=150, edgecolors='grey', norm=norm)
    plt.colorbar(sc, label=r"$|\lambda_B|^2$")
    plt.xlabel(r"$k_B$")
    plt.ylabel(r"$\omega_B$")
    plt.title(f"Schmidt weights p={p:.2f}, E={best_energy:.3f}, L={L}, U={U},N={N}, g={g},")
    plt.tight_layout()
    plt.show()

    return triplets


w_window = (-6,-5) #3body state
w_window = (-3,-0.5) #2body state
w_window = (0,2) #main state
w_window = (2.5,3.5) #main state
#w_window = (0,2.5) #2body state
#lambda_sq, qvals, w_list = schmidt_table(-np.pi, w_window, U, n, g, L, SSB_lambda)
lambda_sq, qvals, w_list = schmidt_table(0, w_window, U, n, g, L, SSB_lambda)



    