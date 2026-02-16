import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import math

a=1
regul =0
   
def R_threshold(chi):
    return 2*np.tan(chi/2)*np.sin(chi/2)
def gamma(R,chi):
    if R <= R_threshold(chi):
        return np.sqrt( np.sin(chi/2)**2 - R**2/4*1/(np.tan(chi/2)**2) )
    else:
        #This allows to recover theta_GS = -pi/4 for the Meissner phase
        return 0
def theta(k,R,chi): 
    x=np.sin(chi/2)*np.sin(chi/2 +k*a -chi*p )/(np.sqrt(R**2/4 + np.sin(chi/2)**2*np.sin(chi/2 +k*a -chi*p)**2))
    x = np.clip(x, -1, 1)
    return -0.5*np.arccos(x)



def normalized(psis,N):
    sum = 0
    for psi in psis:
        sum += abs(psi)**2
    new_psis = [psi*np.sqrt(N/sum) for psi in psis]
    return new_psis

def RK4(f,t0,y0,step,n_steps,J_parallel,J_perp,chi,U,N):
    normalized_y0 = normalized(y0,N)
    t_values = [t0]
    y_values = [normalized_y0]
    for i in range(1,n_steps):
        t_values.append(t_values[-1] + step)
        k1 = f(y_values[-1],t_values[-1],J_parallel,J_perp,U,chi)
        k2 = f(y_values[-1] +step/2*k1, t_values[-1] + step/2 ,J_parallel,J_perp,U,chi)
        k3 = f(y_values[-1] +step/2*k2, t_values[-1] + step/2 ,J_parallel,J_perp,U,chi)
        k4 = f(y_values[-1] +step*k3, t_values[-1] + step ,J_parallel,J_perp,U,chi)
        new_y_value = y_values[-1] + step/6*(k1 +2*k2 +2*k3 +k4)
        normalized_new_y_value = normalized(new_y_value,N)
        y_values.append(np.array(normalized_new_y_value))
    return np.array(t_values), np.array(y_values)


def differential_equations(y,t,J_parallel,J_perp,U,chi) : 
    """ y is a row vector of the form psi_{1,1}, psi_{1,2}, psi_{2,1}, psi_{2,2}, psi_{3,1}, psi_{3,2},... """
    y = np.array(y)
    L = len(y)//2
    psi = y.reshape((L, 2))
    dydt = np.zeros_like(psi, dtype=complex)
    psi_left = np.roll(psi, shift=1, axis=0)
    psi_right = np.roll(psi, shift=-1, axis=0)
    dydt[:, 0] = +J_parallel*(np.exp(-1j*chi*(1-p))*psi_left[:, 0] + np.exp(1j*chi*(1-p))*psi_right[:, 0]) + J_perp*psi[:, 1] - (U/2)*(2*abs(psi[:, 0])**2 ) * psi[:, 0]
    dydt[:, 1] = +J_parallel*(np.exp(1j*chi*p)*psi_left[:, 1]+np.exp(-1j*chi*p)*psi_right[:, 1]) + J_perp * psi[:, 0] - (U / 2)*(2*abs(psi[:, 1])**2)*psi[:, 1]
    return dydt.flatten() 

def Energy_functional(psis,J_parallel,J_perp,chi,U,L):
    psis = psis.reshape(L,2)
    psis_right = np.roll(psis, shift=-1, axis=0)
    kin1 = -J_parallel*(np.exp(1j*chi*(1-p))*np.conjugate(psis[:, 0])@psis_right[:, 0]+np.exp(-1j*chi*(1-p))*psis[:, 0]@np.conjugate(psis_right[:, 0]))
    kin2 = -J_parallel * (np.exp(-1j*chi*p)*np.conjugate(psis[:, 1])@psis_right[:, 1] + np.exp(1j*chi*p)*psis[:, 1]@np.conjugate(psis_right[:, 1]))
    kin3 = -J_perp * (np.conjugate(psis[:, 0]) @ psis[:, 1] + np.conjugate(psis[:, 1]) @ psis[:, 0])
    #interaction = (U/2)*(np.sum(abs(psis[:, 0])**4) *2  + np.sum(abs(psis[:, 1])**4) *2 )
    interaction = (U/2)*(np.sum(abs(psis[:, 0])**4)   + np.sum(abs(psis[:, 1])**4)  )
    summation = kin1 + kin2 + kin3 + interaction
    return summation.real


def Kinetic(psis,J_parallel,J_perp,chi):
    L = len(psis)//2
    psis = psis.reshape(L,2)
    psis_right = np.roll(psis, shift=-1, axis=0)
    kin1 = -J_parallel*(np.exp(1j* chi*(1-p))*np.conjugate(psis[:, 0])@psis_right[:, 0]+np.exp(-1j*chi*(1-p))*psis[:, 0]@np.conjugate(psis_right[:, 0]))
    kin2 = -J_parallel * (np.exp(-1j*chi*p)*np.conjugate(psis[:, 1])@psis_right[:, 1] + np.exp(1j*chi*p)*psis[:, 1]@np.conjugate(psis_right[:, 1]))
    kin3 = -J_perp * (np.conjugate(psis[:, 0]) @ psis[:, 1] + np.conjugate(psis[:, 1]) @ psis[:, 0])
    sum = kin1 + kin2 + kin3
    return sum.real

def Interac(psis,J_parallel,J_perp,U):
    L = len(psis)//2
    psis = psis.reshape(L,2)
    interaction = (U/2)*((sum(abs(psis[:, 0])**4)  ) + (sum(abs(psis[:, 1])**4) ))
    return interaction.real

def chemical_potential(psis,J_parallel,J_perp,chi,U):
    nb_bosons = round((np.conj(psis)@ psis).real,0)
    return (Kinetic(psis,J_parallel,J_perp,chi) + 2* Interac(psis,J_parallel,J_perp,U))/nb_bosons + regul

def psi_mix_k(k,j,m,R,chi,L,omega,phi): #for different k in symetric gauge
    if k ==0:
        omega=0
    real1 = 1/np.sqrt(L)*np.cos( theta(k,R,chi) +np.pi/2*(2-m))*np.cos(omega)*np.cos(j*k*a)
    real2 = 1/np.sqrt(L)*np.cos(theta(-k,R,chi) +np.pi/2*(2-m))* np.sin(omega)*np.cos(-j*k*a + phi)
    imaginary1 = 1/np.sqrt(L)*np.cos(theta(k,R,chi) +np.pi/2*(2-m))*(np.cos(omega)*np.sin(j*k*a) )
    imaginary2 = 1/np.sqrt(L)*np.cos(theta(-k,R,chi) +np.pi/2*(2-m))*(np.sin(omega)*np.sin(-j*k*a + phi))
    return real1 +real2 +1j*(imaginary1 +imaginary2)

def compare_energy(cond1,omega1,cond2,omega2,R,chi,U,L,N,t0,step,n_steps):
    J_parallel = 1
    J_perp = R
    t_values1, y_values1 = RK4(differential_equations, t0, cond1, step, n_steps, J_parallel, J_perp,chi, U,N)
    t_values2, y_values2 = RK4(differential_equations, t0, cond2, step, n_steps, J_parallel, J_perp,chi, U,N)
    final_state1 = y_values1[-1]
    final_state2 = y_values2[-1]
    Energy1 = Energy_functional(final_state1,J_parallel,J_perp,chi,U,L)
    Energy2 = Energy_functional(final_state2,J_parallel,J_perp,chi,U,L)
    if Energy1<Energy2:
        return omega1,cond1,final_state1
    return omega2,cond2,final_state2

def initial_cond_k(k,omega,fluc_max,R,chi, L,phi=0):
    if k==0:
        omega=0
    y0 = np.zeros(2 * L, dtype=complex)
    for site in range(L):
        fluc1 = (np.random.rand() - 0.5)*2*fluc_max
        fluc2 = (np.random.rand() - 0.5)*2*fluc_max
        y0[2*site] = psi_mix_k(k,site, 1, R, chi, L, omega, phi) + fluc1
        y0[2*site+1] = psi_mix_k(k,site, 2, R, chi, L, omega, phi) + fluc2
    return omega ,y0


def GPE_condensate(R,chi,U,p,L,N,J_parallel =1,fluc_max = 0, phi =0,t0=0,step = 0.01,n_steps = 1000):
    time_start = time.time()
    J_perp = R
    energies = []
    final_states = []
    #for i in range(-L//2,L//2):
    for i in range(0,L//2):
    
        omega1 = np.pi/4
        omega2 = 0   # So i'm selecting GS +
        omega1, cond1 = initial_cond_k(2*np.pi*i/L,omega1,fluc_max,R,chi, L,phi)
        omega2, cond2 = initial_cond_k(2*np.pi*i/L,omega2,fluc_max,R,chi, L,phi)
        #print('momentum of initial cond on leg 1',(np.angle(cond2[0]   * np.conj(cond2[2])))*L/(2*np.pi) )  # shoudl be minus
        omega,y0,final_state = compare_energy(cond1,omega1,cond2,omega2,R,chi,U,L,N,t0,step,n_steps)
        final_states = final_states + [[i,omega,final_state]]
        E =  Energy_functional(final_state,J_parallel,J_perp,chi,U,L)
        energies = energies + [E]
    best_state_index = np.argmin(energies)
    print('GPE condensate computation took', time.time()-time_start)
    return final_states[best_state_index]


def T_matrix_canonical_basis(pow,L): # For this condensate to not break translational invariance the period should be dividing L
    A1 = np.zeros((L, L), dtype=complex)
    zero = np.zeros_like(A1)
    phase_factor = np.exp(-1j*pow*b*2*np.pi/L) if c != np.pi/4 else 1  # because the condensate has a phase of PLUS 2*b*np.pi/L
    phase_factor = np.exp(-1j*pow*b*2*np.pi/L)
    #phase_factor = -1
    for k in range(L):
        A1[k,(k+pow)%L] = + 1
    return np.block([[phase_factor*A1, zero ,zero , zero],[zero , phase_factor*A1 , zero, zero ], [zero , zero,  np.conj(phase_factor)*A1, zero ],[zero, zero, zero,  np.conj(phase_factor)*A1] ])
    #return np.block([[1*A1, zero ,zero , zero],[zero , 1*A1 , zero, zero ], [zero , zero,  -A1, zero ],[zero, zero, zero,  -A1] ])

def A_m_Real( J_parallel,J_perp, U,chi, psi_GS,phi_m,m):
    L = len(psi_GS)//2
    Am = np.zeros((L, L), dtype=complex)
    psi_GS_reshaped = np.reshape(psi_GS, (L,2))
    for j in range(L):
        Am[j,j] = -2*U*abs(psi_GS_reshaped[j,m])**2 + 1*chemical_potential(psi_GS,J_parallel,J_perp,chi,U)
        Am[j,(j+1)%L] = J_parallel*np.exp(1j*phi_m)                     
        Am[j,(j-1)%L] = J_parallel*np.exp(-1j*phi_m)                   
    return Am

def A_m_Real_hole(J_parallel,J_perp, U, chi,psi_GS,phi_m,m):
    L = len(psi_GS)//2
    Am = np.zeros((L, L), dtype=complex)
    psi_GS_reshaped = np.reshape(psi_GS, (L,2))
    for j in range(L):
        Am[j,j] = -2*U*abs(psi_GS_reshaped[j,m])**2 + 1*chemical_potential(psi_GS,J_parallel,J_perp,chi,U)
        Am[j,(j+1)%L] = J_parallel*np.exp(1j*phi_m)                    
        Am[j,(j-1)%L] = J_parallel*np.exp(-1j*phi_m)                   
    return Am
def B_m_Real( U, psi_GS,m):
    L = len(psi_GS)//2
    Bm = np.zeros((L, L), dtype=complex)
    psi_GS_reshaped = np.reshape(psi_GS, (L,2))
    for j in range(L):
        Bm[j,j] = -U*psi_GS_reshaped[j,m]**2
    return Bm

def C_m_Real(psi_GS, J_perpendicular):
    L = len(psi_GS)//2
    C = np.zeros((L, L), dtype=complex)
    for j in range(L):
        C[j,j] = J_perpendicular
    return C

def construct_L(A1, B1, A2, B2, C1, C2,A1_hole,A2_hole, L,pow =1, epsilon= 0.00001):
    Translation = T_matrix_canonical_basis(pow,L)
    zero = np.zeros_like(A1)
    return np.block([[-A1, -C1, -B1, zero],[ -C2, -A2, zero, -B2], [B1.conj(), zero, A1_hole.conj(), C1.conj()],[zero, B2.conj(), C2.conj(), A2_hole.conj()] ]) + 2*epsilon*(Translation - np.conj(Translation).T)
    #return np.block([[-A1, -C1, -B1, zero],[ -C2, -A2, zero, -zero], [B1.conj(), zero, A1_hole.conj(), C1.conj()],[zero, zero, C2.conj(), A2_hole.conj()] ]) + 2*1j*epsilon*(Translation - np.conj(Translation).T)
    #return np.block([[A1, C1, B1, zero],[ C2, A2, zero, B2], [-B1.conj().T, zero, -A1_hole.conj().T, -C1.conj().T],[zero, -B2.conj().T, -C2.conj().T, -A2_hole.conj().T] ]) + 2*1j*epsilon*(Translation - Translation.T)

def symplectic_normalization_ladder(uv_matrix): # 2L-1 columns and 4L rows 
    u_values = uv_matrix[:2*L,:]
    v_values = np.conj(uv_matrix[2*L:,:])
    #symp_norm = np.tile(np.sqrt(  np.sum( np.abs(u_values)**2) - np.sum(np.abs(v_values)**2, axis = 0)  ),(2*L,1))
    symp_norm = np.sqrt(  np.sum( np.abs(u_values)**2,axis = 0) - np.sum(np.abs(v_values)**2, axis = 0)  )
    print('output', np.sqrt(np.sum(np.abs( (u_values/symp_norm[np.newaxis, :])[:,0])**2) - np.sum(np.abs((v_values/symp_norm[np.newaxis, :])[:,0])**2) ) )
    return u_values/symp_norm[np.newaxis, :],  v_values/symp_norm[np.newaxis, :]





#########################################################################################################################################################################################
########################################################################## Dynamical structure factor : bath ############################################################################
#########################################################################################################################################################################################


J_parallel = 1
J_perp = 2.2
#J_perp = 1.2
#J_perp = 0.25
#J_perp = 0.1
R = J_perp/J_parallel
U = 0.2
p = 1/2
chi = np.pi*0.996
chi = np.pi*0.66
chi = np.pi/2
L=4*24
L= 32
L=100
L=96*2
L=80
L=30
N=L
Gamma = 0.05
b,c,condensate =GPE_condensate(R,chi,U,p,L,N,J_parallel, 0, 0,0, 0.01,40)

#condensate[::2], condensate[1::2] = condensate[1::2], condensate[::2]


pow =   L//math.gcd(2*b, L) if c == np.pi/4 else 1 
print('U,R,chi =', U,R,chi)
print("N=", N)
print('norm =', np.conj(condensate)@condensate)
print('b, omega,POW =', b,c,pow)

A1_R = A_m_Real(J_parallel, J_perp,U, chi,condensate,chi*(1-p),0)

A2_R = A_m_Real( J_parallel, J_perp,U,chi, condensate,-chi*p,1)
B1_R = B_m_Real( U, condensate,0)
B2_R = B_m_Real(U, condensate,1)
C1_R = C_m_Real(condensate, J_perp)
C2_R = C1_R
A1_R_hole = A_m_Real_hole(J_parallel,J_perp, U,chi, condensate,chi*(1-p),0)
A2_R_hole = A_m_Real_hole(J_parallel,J_perp, U, chi,condensate,-chi*p,1)

eom_matrix = construct_L(A1_R, B1_R, A2_R, B2_R, C1_R, C2_R,A1_R_hole,A2_R_hole,L,pow,0.0000000001)

eom_eigenvalues_unsorted, eom_eigenvectors_unsorted = np.linalg.eig(eom_matrix )
sorted_indices = np.argsort(eom_eigenvalues_unsorted)

Bogo_spec, Bogo_eigenvectors = eom_eigenvalues_unsorted[sorted_indices][1+2*L:].real, eom_eigenvectors_unsorted[:, sorted_indices][:, 1+2*L:]

#u_matrix,v_matrix = symplectic_normalization_ladder(Bogo_eigenvectors[:,2*L+1:])

q_vals = 2*np.pi/L*np.arange(L) -np.pi 
Omega_vals = np.linspace(-1,7.2,800)
Omega_vals = np.linspace(-0.1, 8,800)
m=1
m_prime = 1
def dyn_structure_factor(q_vals,Omega_vals,m,m_prime,Bogo_spec,condensate,Bogo_eigenvectors,Gamma = 0.25):
    L = len(condensate)//2
    condensate_reshaped =  np.reshape(condensate, (L,2))
    phi_star_j_m = np.conj(condensate_reshaped[:,m])[:,None,None,None] #index j mu q omega
    phi_j_m_prime = condensate_reshaped[:,m_prime][:,None,None,None]
    
    u_unnormalized = Bogo_eigenvectors[:2*L,:]
    v_unnormalized = np.conj(Bogo_eigenvectors[2*L:,:])  

    symplectic_norm = np.sqrt((np.abs(u_unnormalized)**2 - np.abs(v_unnormalized)**2)*2*L)
    u,v = u_unnormalized/ symplectic_norm, v_unnormalized/symplectic_norm
    print('symp norm',np.sqrt((np.abs(u)**2 - np.abs(v)**2)*2*L) )
    tolerance = 1e-10
    bad_values = (np.abs(u_unnormalized)**2 - np.abs(v_unnormalized)**2)*2*L

    vj_mprime_nu = v.reshape(2, L, -1)[m_prime,:,:][:,:,None,None] 
    u_star_j_m_nu = np.conj(u).reshape(2, L, -1)[m,:,:][:,:,None,None] 
    phase_factor =np.exp(1j*q_vals[None,None,:,None]*np.arange(L)[:,None,None,None])
    denominator = (Omega_vals[None,None,:] -Bogo_spec[:,None,None] + 1j*Gamma)
    super_oscillator_strength = phase_factor*(  phi_star_j_m * vj_mprime_nu + phi_j_m_prime * u_star_j_m_nu )
    oscillator_strength = np.abs(np.sum(super_oscillator_strength ,axis = (0) ))**2
    spec_matrix = np.sum( oscillator_strength/denominator,axis =0 )
    return -2/L**2*spec_matrix.imag

def dyn_structure_factor(q_vals, Omega_vals, m, m_prime, Bogo_spec, condensate, Bogo_eigenvectors, Gamma=0.25):
    L = len(condensate) // 2
    condensate_reshaped = np.reshape(condensate, (L, 2))
    phi_star_j_m = np.conj(condensate_reshaped[:, m])[:, None, None, None]  # index j, m, q, omega
    phi_j_m_prime = condensate_reshaped[:, m_prime][:, None, None, None]

    phi_star_j_1 = np.conj(condensate_reshaped[:, 1])[:, None, None, None]  # index j, m, q, omega
    phi_j_1 = condensate_reshaped[:, 1][:, None, None, None]
    phi_star_j_2 = np.conj(condensate_reshaped[:, 1])[:, None, None, None]  # index j, m, q, omega
    phi_j_2 = condensate_reshaped[:, 1][:, None, None, None]
    u_values, v_values = symplectic_normalization_ladder(Bogo_eigenvectors)
    #print('Check symplectic norm', np.sqrt(   np.sum( np.abs(u_values[:,0])**2) - np.sum(np.abs(v_values[:,0])**2))      )
    print('Check symplectic norm', np.sqrt(   np.sum( np.abs(u_values[:,0])**2-  np.abs(v_values[:,0])**2))      )
    vj_mprime_nu = v_values.reshape(2, L, -1)[m_prime, :, :][:, :, None, None]
    u_star_j_m_nu = np.conj(u_values).reshape(2, L, -1)[m, :, :][:, :, None, None]
    vj_1_nu = v_values.reshape(2, L, -1)[0, :, :][:, :, None, None]
    u_star_j_1_nu = np.conj(u_values).reshape(2, L, -1)[0, :, :][:, :, None, None]
    vj_2_nu = v_values.reshape(2, L, -1)[1, :, :][:, :, None, None]
    u_star_j_2_nu = np.conj(u_values).reshape(2, L, -1)[1, :, :][:, :, None, None]
    phase_factor = np.exp(1j * q_vals[None, None, :, None] * np.arange(L)[:, None, None, None])
    denominator = (Omega_vals[None, None, :] - Bogo_spec[:, None, None] + 1j * Gamma)
    super_oscillator_strength = phase_factor * (phi_star_j_m * vj_mprime_nu + phi_j_m_prime * u_star_j_m_nu)
    #super_oscillator_strength = phase_factor * (phi_star_j_1 * vj_1_nu + phi_j_1 * u_star_j_1_nu + phi_star_j_2 * vj_2_nu + phi_j_2 * u_star_j_2_nu)
    oscillator_strength = np.abs(np.sum(super_oscillator_strength, axis=0))**2
    spec_matrix = np.sum(oscillator_strength / denominator, axis=0)
    return -2 / L**2 * spec_matrix.imag

print('dyn str factor', dyn_structure_factor(q_vals,Omega_vals,m,m_prime,Bogo_spec,condensate,Bogo_eigenvectors,Gamma ) )
def dyn_structure_factor_bonding(q_vals, Omega_vals, s, s_prime, Bogo_spec, condensate, Bogo_eigenvectors, Gamma=0.25):
    L = len(condensate) // 2
    condensate = condensate.reshape(L, 2)
    k_vals = 2 * np.pi * np.arange(L) / L - np.pi
    theta_k = 0.5 * np.arctan2(
        np.imag(condensate[:,1] * np.conj(condensate[:,0])),
        np.real(condensate[:,0] * np.conj(condensate[:,0])))
    zeta_1 = np.cos(theta_k[:, None] + q_vals[None, :] - np.pi/2 * (1 - s)) * np.cos(theta_k[:, None] - np.pi/2 * (1 - s_prime))
    zeta_2 = np.cos(theta_k[:, None] + q_vals[None, :] - np.pi/2 * (1 - s)) * np.sin(theta_k[:, None] - np.pi/2 * (1 - s_prime))
    zeta_3 = np.sin(theta_k[:, None] + q_vals[None, :] - np.pi/2 * (1 - s)) * np.cos(theta_k[:, None] - np.pi/2 * (1 - s_prime))
    zeta_4 = np.sin(theta_k[:, None] + q_vals[None, :] - np.pi/2 * (1 - s)) * np.sin(theta_k[:, None] - np.pi/2 * (1 - s_prime))
    phi_1 = condensate[:, 0]
    phi_2 = condensate[:, 1]
    uj = Bogo_eigenvectors[:2*L, :].reshape(2, L, -1)  
    vj = np.conj(Bogo_eigenvectors)[2*L:, :].reshape(2, L, -1) 
    M_total = np.zeros((len(q_vals), uj.shape[2]), dtype=np.complex128)
    for j in range(L):
        phase_j = np.exp(1j * q_vals * j)  
        for k in range(L):
            terms = [ phi_1[j].conj() * vj[0, k] + phi_1[k] * np.conj(uj[0, j]), phi_1[j].conj() * vj[1, k] + phi_2[k] * np.conj(uj[0, j]), phi_2[j].conj() * vj[0, k] + phi_1[k] * np.conj(uj[1, j]),phi_2[j].conj() * vj[1, k] + phi_2[k] * np.conj(uj[1, j]), ] 
            for i, zeta in enumerate([zeta_1[k], zeta_2[k], zeta_3[k], zeta_4[k]]):
                M_total += zeta[:, None] * phase_j[:, None] * terms[i][None, :]  
    numerator = np.abs(M_total)**2  
    denominator = Omega_vals[None, :] - Bogo_spec[:, None] + 1j * Gamma  
    spectral_values = np.sum(numerator[:, :, None] / denominator[None, :, :], axis=1)  

    return -2.0 / L * spectral_values.imag

""" current density perturbation... """
def dyn_structure_factor_current(q_vals, Omega_vals, m, m_prime, Bogo_spec, condensate, Bogo_eigenvectors, Gamma=0.25):
    L = len(condensate) // 2
    condensate_reshaped = condensate.reshape(L, 2)
    phi_1 = condensate_reshaped[:, 0][:, None]  
    phi_2 = condensate_reshaped[:, 1][:, None]
    u = Bogo_eigenvectors[:2*L, :].reshape(2, L, -1) # shape (2, L,  2L -1)
    v = np.conj(Bogo_eigenvectors[2*L:, :].reshape(2, L, -1))
    phi_1_next = np.roll(phi_1, -1, axis=0)           # shape (L,1)
    phi_2_next = np.roll(phi_2, -1, axis=0)           # shape (L,1)
    v_next   = np.roll(v[:,:,:], -1, axis=2)       # shape (L, N_modes)
    u_next   = np.roll(u[:,:,:], -1, axis=2)       # shape (L, N_modes)
    print("u shape", u.shape)
    S_qw = np.zeros((len(q_vals), len(Omega_vals)), dtype=np.complex128)

    for nu in range(Bogo_spec.size):
        phase_factor = np.exp(1j * q_vals[None, :] * np.arange(L)[:, None])  
        # J perp at j
        term_J_perp = ( np.conj(phi_2) * v[0, :, nu][:, None] +
             phi_1 * np.conj(u[1, :, nu])[:, None] -
            np.conj(phi_1) * v[1, :, nu][:, None] -
            phi_2 * np.conj(u[0, :, nu])[:, None] )
        # J perp at j+1
        term_J_perp_next = ( np.conj(phi_2_next) * v_next[0, :, nu][:, None] +
            phi_1_next * np.conj(u_next[1, :, nu])[:, None] -
            np.conj(phi_1_next) * v_next[1, :, nu][:, None] -
            phi_2_next * np.conj(u_next[0, :, nu])[:, None] )
        # J parallel 1
        term_J_parallel_1 = (
        np.exp(1j*chi/2) * ( np.conj(phi_1_next) * v[0,:,nu][:, None] + phi_1 * np.conj(u[0,:,nu][:, None]) ) 
        - np.exp(-1j*chi/2) * ( np.conj(phi_1) * v[0,:,nu][:, None] + phi_1_next * np.conj(u[0,:,nu][:, None]) ))
        #J parallel 2
        term_J_parallel_2 = (
        np.exp(-1j*chi/2) * ( np.conj(phi_2_next) * v[1,:,nu][:, None] + phi_2 * np.conj(u[1,:,nu][:, None]) ) 
        - np.exp(1j*chi/2) * ( np.conj(phi_2) * v[1,:,nu][:, None] + phi_2_next * np.conj(u[1,:,nu][:, None]) ))
        #term = term_J_parallel_1
        #term = term_J_parallel_1 + term_J_perp - term_J_parallel_2 - term_J_perp_next
        term =  term_J_perp
        super_osc_strength = np.sum(phase_factor * term, axis=0)  # (len(q_vals),)
        denom = Omega_vals - Bogo_spec[nu] + 1j*Gamma  # (len(Omega_vals),)
        S_qw += (np.abs(super_osc_strength)[:, None]**2) / denom[None, :]
    return -2 / L**2 * S_qw.imag



def dyn_structure_factor_plot(f, q_vals, Omega_vals, m, m_prime, Bogo_spec, condensate, Bogo_eigenvectors, R,U,L,Gamma=0.25):
    k_values = []
    energy_values = []
    spectral_values = f(q_vals, Omega_vals, m, m_prime, Bogo_spec, condensate, Bogo_eigenvectors, Gamma)
    Q_mesh, Omega_mesh = np.meshgrid(q_vals, Omega_vals, indexing='ij')
    norm = mcolors.LogNorm(vmin=max(spectral_values.min(), 2e-4), vmax=spectral_values.max() )
    plt.pcolormesh(Q_mesh, Omega_mesh, spectral_values, cmap='viridis', norm=norm, shading='auto')
    cbar = plt.colorbar(label=r"$S(q,\Omega)$")
    cbar.ax.tick_params(labelsize=25) 
    cbar.set_label(r"$S(q,\Omega)$", fontsize=20)  
    plt.xlabel(r"Momentum $q$",size = 20)
    plt.ylabel(r"Frequency $\Omega$",size = 20)
    #plt.title(f"Dynamical Structure Factor for m={m+1}, R={R}, U={U}," + r"$\chi$= " +f"{round(chi/np.pi,2)}"+r"$\pi, \Gamma =$"+f'{Gamma}, L={L}',size = 15) 
    #plt.title(  f"Dynamical Structure Factor for m={m+1}, R={R}, U={U}, "  + r"$\chi$= " + f"{round(chi/np.pi, 2)}" + r"$\pi, \Gamma =$" + f"{Gamma}, L={L}",  size=15,pad=20 )
    plt.title(  f"Dynamical current response m={1}, R={R}, U={U}, "  + r"$\chi$= " + f"{round(chi/np.pi, 2)}" + r"$\pi, \Gamma =$" + f"{Gamma}, L={L}",  size=15,pad=20 )
    #plt.title(f"Dynamical Structure Factor for $R={R}, U={U},$" + r'$\chi=\frac{\pi}{2}, \Gamma =$'+f'{Gamma}, L={L}', size = 15)
    #plt.xlim(-1,1)
    #plt.ylim(-1,1)
    plt.xticks(size=15)
    plt.yticks(size=20)
    plt.tight_layout()
    plt.show()

#S_perp = dyn_structure_factor_perp(q_vals, Omega_vals, m, m_prime, Bogo_spec, condensate, Bogo_eigenvectors, Gamma)

##dyn_structure_factor_plot( dyn_structure_factor_current,   q_vals,Omega_vals, m=0,   m_prime=0,Bogo_spec=Bogo_spec,condensate=condensate, Bogo_eigenvectors=Bogo_eigenvectors, R=R, U=U,L=L, Gamma=Gamma)


#dyn_structure_factor_plot(dyn_structure_factor, q_vals, Omega_vals, m, m_prime, Bogo_spec, condensate, Bogo_eigenvectors,R,U,L,Gamma)

#dyn_structure_factor_plot(dyn_structure_factor_bonding, q_vals, Omega_vals, m, m_prime, Bogo_spec, condensate, Bogo_eigenvectors,R,U,L,Gamma)


#########################################################################################################################################################################################
########################################################################## Current pattern : excited states ############################################################################
#########################################################################################################################################################################################


def AB(m, m_prime, j, mu1, mu2, Bogo_eigenvectors):
    """
    Compute A^{mm'j}_{mu1,mu2} and B^{mm'j}_{mu1,mu2}.
    """
    u = Bogo_eigenvectors[:2*L, :].reshape(2, L, -1)
    v = np.conj(Bogo_eigenvectors[2*L:, :].reshape(2, L, -1))
    A = np.conj(u[m-1, j, mu1]) * u[m_prime-1, j, mu2]
    B = np.conj(v[m-1, j, mu1]) * v[m_prime-1, j, mu2]
    return A, B

def J_perp_excited(nu, Bogo_eigenvectors, R):
    """
    Compute <nu|J_perp_j|nu> for all j.
    """
    J_perp = np.zeros(L, dtype=float)
    mu_range = np.arange(2*L - 1)

    for j in range(L):
        # diagonal terms for this nu
        A21_nunu, B21_nunu = AB(2, 1, j, nu, nu, Bogo_eigenvectors)
        # trace over mu
        B21_trace = 0
        for mu in mu_range:
            _, B21_mu = AB(2, 1, j, mu, mu, Bogo_eigenvectors)
            B21_trace += B21_mu.imag

        J_perp[j] = -R*(
            2*A21_nunu.imag
            + 2*B21_nunu.imag
            + 2*B21_trace)

    plt.plot(np.arange(L), J_perp, marker='o')
    plt.xlabel('j')
    plt.ylabel(r'$\langle \nu|J^\perp_{j,1\to 2}|\nu\rangle$')
    plt.title(rf'Perpendicular current in excited state $\nu$ ={nu}')
    plt.grid()
    plt.show()
    return J_perp

J_perp_excited(1, Bogo_eigenvectors, R)