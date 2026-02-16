from Chevy_1D_polaron_spec import *
from qutip import destroy, qeye, tensor
import numpy as np


L = 4          # number of sites
Nmax = 4       # local Hilbert space cutoff
J = 1.0        # hopping
U = 1.0        # on-site interaction
N = 4          # total particle number.... it doesnt really make sense since I am not number conserving but let's say it's the average
n = N/L      # condensate density
lam = 0.1      # symmetry-breaking
mu = 2 + U*N/L + lam/np.sqrt(N)    # chemical potential


gamma = [tensor([destroy(Nmax) if i==j else qeye(Nmax) for i in range(L)])
         for j in range(L)]


qvals = 2*np.pi/L * np.arange(L)


a_k = np.zeros(L,dtype =object)
for idx, k in enumerate(qvals):
    ak = 0
    for j in range(L):
        ak += uk_eta(k, U, n, lam) * np.exp(1j*k*j) * gamma[j]
        ak += vk_eta(k, U, n, lam) * np.exp(-1j*k*j) * gamma[j].dag()

    if idx == 0:
        ak += np.sqrt(N)
    a_k[idx] = ak
H = 0
for idx, k in enumerate(qvals):
    eps = -2*J*np.cos(k) - mu
    H += eps * a_k[idx].dag() * a_k[idx]


for k1 in range(L):
    for k2 in range(L):
        for k3 in range(L):
            k4 = (k1+k2-k3) % L
            H += (U/(2*L)) * a_k[k1].dag() * a_k[k2].dag() * a_k[k3] * a_k[k4]


H += lam * (a_k[0] + a_k[0].dag())

print("Hamiltonian in γ_j basis:")
print(H)
