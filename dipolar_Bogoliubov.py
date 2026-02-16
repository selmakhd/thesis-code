import numpy as np
import matplotlib.pyplot as plt
import cmath


L = 20          # number of sites
V = 2.7187  #polar interaction strength for coulomb
#V= 0.973    #good for softcore
#V= 2.8      #good for softened dipolar
#V=1
V=0.5
N = 40         # condensate density (number per site)
t = 1.0            # hopping amplitude (for epsilon_k = -2t cos k)
n0 = N/L
alpha = 3
#alpha = 1
delta = 1.5
#delta =0
softcore = 3 #width of the softcore
# Allowed momenta (PBC)
k_vals = 2*np.pi*(np.arange(L)-L//2)/L 

def f(r, L):
    return min(r, L - r)


#softcore potential
Vq = np.zeros(L)
for i, k in enumerate(k_vals):
    sumV = 0.0
    for r in range(1,softcore +1):
        weight = 1.0
        sumV += weight * np.cos(k*r)
    Vq[i] = V*sumV

# Dipolar potential Fourier transform
Vq = np.zeros(L)
for i, k in enumerate(k_vals):
    sumV = 0.0
    for r in range(1, L//2 + 1):
        weight = 1.0
        if L%2== 0 and r==L//2:
            weight = 0.5  # special term for opposite site
        sumV += weight*np.cos(k*r)/(r**alpha+ delta**alpha)
    Vq[i] = V/(n0**alpha)*sumV


#Contact interaction
Vq = np.ones(L)*V
#print('Interaction in momentum space', Vq)

#Noninteracting dispersion
V0 =Vq[L//2]
epsilon0 = -2*t
mu = epsilon0 + n0*V0    # chemical potential
Ek = -2*t*np.cos(k_vals) - mu


# Cartoonish spectrum M shape
def dispersion(p1,p2,p3,r1,r2,L):
    #p1 p2 p3 are slopes and p2 should be negative
    #r1 is the position of the maximum and r2 the mimumum
    k_pos = np.linspace(0, np.pi, L//2)
    omega_pos = np.zeros_like(k_pos)
    for i, k in enumerate(k_pos):
        if k <= r1:
            omega_pos[i] = p1 * k
        elif k <= r2:
            omega_pos[i] = p1*r1 + p2*(k - r1)
        else:
            omega_pos[i] = p1*r1 + p2*(r2 - r1) + p3*(k - r2)

    # Symmetrize to negative k
    k_full = np.concatenate([-k_pos[-2::-1], k_pos])
    omega_full = np.concatenate([omega_pos[::-1], omega_pos])
    return omega_full



#Bogoliubov spectrum
Fk = (Ek+(2*Vq +V0)*n0 )*(Ek + n0*V0)
print('min of Fk', np.min(Fk))
omega_k = np.sqrt(np.abs((Ek+(2*Vq +V0)*n0 )*(Ek + n0*V0)))
omega_k = np.sqrt(Fk.astype(complex)).real
p1,p2,p3 = 1,-1,1
r1,r2 = np.pi/L*20, np.pi/L*30
#omega_k = dispersion(p1,p2,p3,r1,r2,L)
omega_k_r = np.imag(omega_k)
omega_k_i = np.real(omega_k)

plt.figure(figsize=(6,4))
plt.plot(k_vals, omega_k_r, '-')
plt.plot(k_vals, omega_k_i, '-')
#plt.plot(k_vals, Vq, '-')
plt.xlabel('k')
plt.ylabel('Bogoliubov spectrum ω_k')
plt.title(f'1D Dipolar Bogoliubov Spectrum, L = {L},V={V},alpha={alpha}, delta={delta}')
plt.grid(True)
plt.show()
