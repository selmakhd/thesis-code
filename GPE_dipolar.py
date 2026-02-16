import numpy as np
import time
import matplotlib.pyplot as plt
L = 12            # number of sites
V = 1.6           # dipolar interaction strength
U= 0           # to enforce hardcore boson condition
N = 4            # condensate density (number per site)
t = 1.0           # hopping amplitude (for epsilon_k = -2t cos k)
n0 = N/L
# Allowed momenta (PBC)
k_vals = 2*np.pi*(np.arange(L)-L//2)/L 
def f(r, L):
    return min(r, L - r)

def normalized(psis,N):
    sum = 0
    for psi in psis:
        sum += abs(psi)**2
    new_psis = [psi*np.sqrt(N/sum) for psi in psis]
    return new_psis

def RK4(f,t0,y0,step,n_steps,J,V,U,N):
    normalized_y0 = normalized(y0,N)
    #print('normalized y0 is', np.linalg.norm(normalized_y0)  )
    t_values = [t0]
    y_values = [normalized_y0]
    for i in range(1,n_steps):
        t_values.append(t_values[-1] + step)
        k1 = f(y_values[-1],t_values[-1],J,V,U)
        k2 = f(y_values[-1] +step/2*k1, t_values[-1] + step/2 ,J,V,U)
        k3 = f(y_values[-1] +step/2*k2, t_values[-1] + step/2 ,J,V,U)
        k4 = f(y_values[-1] +step*k3, t_values[-1] + step ,J,V,U)
        new_y_value = y_values[-1] + step/6*(k1 +2*k2 +2*k3 +k4)
        normalized_new_y_value = normalized(new_y_value,N)
        y_values.append(np.array(normalized_new_y_value))
    return np.array(t_values), np.array(y_values)

def differential_equations(y,t,J,V,U) : 
    """ y is a row vector of the wavefunction on the lattice"""
    y = np.array(y)
    L = len(y)
    psi = y
    n = np.abs(psi)**2
    psi_left = np.roll(psi, shift=1)
    psi_right = np.roll(psi, shift=-1)
    U_dip = np.zeros(L)
    for j in range(L): #I do not assume anything about broken translation becasue you never know
        s = 0.0
        for l in range(L):
            if l != j:
                r = f(abs(j - l), L)
                s += n[l] / (r**3)
        U_dip[j] = s
    dydt = +J*(psi_left+psi_right)  -V*U_dip * psi/n0**3  - (U/2)*(2*abs(psi)**2 )*psi
    return dydt

def initial_cond_k(k,omega,fluc_max, L,phi=0):
    if k==0:
        omega=0
    y0 = np.zeros( L, dtype=complex)
    y0 = np.array([np.cos(omega)*np.exp(1j*k*i) + np.sin(omega)*np.exp(-1j*k*i) + (np.random.rand() - 0.5)*2*fluc_max for i in range(L) ])
    return omega ,y0/np.linalg.norm(y0)

def Energy_functional(psis,J,V,U,L):
    psis_right = np.roll(psis, shift=-1)
    n = np.abs(psis)**2
    kin = -J*(np.conjugate(psis)@psis_right+psis@np.conjugate(psis_right))
    interaction = 0.0
    for i in range(L):
        for j in range(L):
            if i != j:
                r = f(abs(j-i), L)
                interaction += n[i]*n[j]/(r**3)
    interaction *= V/2/n0**3   
    interaction2 = U/2*np.sum(abs(psis)**4) 
    summation = kin+ interaction + interaction2
    return summation.real

def Kinetic(psis,J):
    psis_right = np.roll(psis, shift=-1)
    return -J*(np.conjugate(psis)@psis_right+psis@np.conjugate(psis_right))
def Interac(psis,V,U):
    n = np.abs(psis)**2
    interaction = 0.0
    for i in range(L):
        for j in range(L):
            if i != j:
                r = f(abs(j-i), L)
                interaction += n[i]*n[j]/(r**3)
    interaction *= V/2/n0**3   
    interaction2 = U/2*np.sum(abs(psis)**4) 
    return interaction + interaction2
def compare_energy(cond1,omega1,cond2,omega2,V,U,L,N,t0,step,n_steps):
    J = 1
    t_values1, y_values1 = RK4(differential_equations, t0, cond1, step, n_steps, J,V,U,N)
    t_values2, y_values2 = RK4(differential_equations, t0, cond2, step, n_steps, J, V,U,N)
    final_state1 = y_values1[-1]
    final_state2 = y_values2[-1]
    Energy1 = Energy_functional(final_state1,J,V,U,L)
    Energy2 = Energy_functional(final_state2,J,V,U,L)
    print('energy 1', Energy1 )
    print('energy 2', Energy2 )
    if Energy1<Energy2:
        return omega1,cond1,final_state1
    return omega2,cond2,final_state2

def GPE_condensate(L,N,V,U,J =1,fluc_max = 0, phi=0,t0=0,step = 0.01,n_steps = 1000):
    time_start = time.time()
    energies = []
    final_states = []
    for i in range(0,L//2):
    #for i in range(0,L):
        print('i is', i)
        omega1 = np.pi/4
        omega2 = 0   #So positive momentum
        omega1, cond1 = initial_cond_k(2*np.pi*i/L,omega1,fluc_max, L,phi)
        omega2, cond2 = initial_cond_k(2*np.pi*i/L,omega2,fluc_max, L,phi)
        omega,y0,final_state = compare_energy(cond1,omega1,cond2,omega2,V,U,L,N,t0,step,n_steps)
        final_states = final_states + [[i,omega,final_state]]
        E =  Energy_functional(final_state,J,V,U,L)
        energies = energies + [E]
    best_state_index = np.argmin(energies)
    print('GPE condensate computation took', time.time()-time_start)
    return final_states[best_state_index]

def GPE_evol(psi0,L,N,V,U,J =1,t0=0,step = 0.01,n_steps = 1000):
    t_values1, y_values1 = RK4(differential_equations, t0, psi0, step, n_steps, J,V,U,N)
    return  t_values1, y_values1

def Energy_plot_GPE(psi0,L,N,V,U,J =1,t0=0,step = 0.001,n_steps = 2000):
    tvals,yvals =GPE_evol(psi0,L,N,V,U,J =1,t0=0,step = 0.001,n_steps = 2000)
    energies =[]
    for yval in yvals:
        energies.append(Energy_functional(yval,1,V,U,L))
    plt.plot(tvals,energies)
    plt.show()

def vortex_density(psis,L):
    psis_real = [abs(psi) for psi in psis]
    mean = sum(psis_real)/len(psis_real)
    modulus = abs(psis[0])
    vortex_number = 0
    for j in range(1,L):
        if (modulus-mean)*(abs(psis[j])-mean) < -0.00000001:
            vortex_number += 1
            modulus = abs(psis[j])
    vortex_density = vortex_number//2/L
    return vortex_density

psi0 = initial_cond_k(np.pi/L,np.pi/4,fluc_max=0, L=L,phi=0)[-1]
#psi0= np.ones(L)*np.sqrt(n0)
#print('vortex density is',vortex_density(psi0,L))
Energy_plot_GPE(psi0,L,N,V,U,J =1,t0=0,step = 0.001,n_steps = 2000)

#state = GPE_condensate(L,N,V,U,J =1,fluc_max = 0, phi=0,t0=0,step = 0.001,n_steps = 5000)[-1]

#print('state is', state)
#print('total number of particles ', np.linalg.norm(np.array(state))**2)
#plt.plot(np.arange(L),np.abs(state))
#plt.show()