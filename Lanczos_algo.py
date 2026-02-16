from scipy.linalg import eigh_tridiagonal, norm
import numpy as np

def lanczos_core(Psi0, H, V, M=100):
    """ Input 
        Psi0 : initial state in a Quspin basis (much larger than M),  
        H : Hamiltonian in Quspin basis, 
        V: a bunch of column vectors expressed in the Krylov basis,
        M : number of Lanczos iterations
        Output
        A = [a_0,...,a_{M}] the diagonal coefficients
        B = [0,b_1,...,b_{M}] the off-diagonal coefficients from wich you build H' within Krylov subspace
        Remark : I assume that they are all linearly independant though... Otherwise the a b coeff are ill-defined
      """
    VV = np.zeros((len(Psi0), V.shape[1]), dtype=Psi0.dtype) # Ns lines et columns equal to number of vectors i wanna convert 
    A = np.zeros(M +1, dtype=Psi0.dtype)
    B = np.zeros(M +1, dtype=Psi0.dtype)
    # I need to keep in memory 3 vectors at a time
    # Initialisation
    phi0 = Psi0.copy() # The only one i need for a_0  and  b_0
    phi0 = phi0/norm(phi0) # The only one i need for a_0  and  b_0
    phi_minus_1 = np.zeros_like(Psi0)
    b1 = norm(phi0) 
    #print('b1 is', b1)
    #phi0 = phi0/b1
    b1 = norm(phi0)
    #b1 = 1
    B[0] = 0
    A[0] = np.vdot(phi0,H.dot(phi0)) / np.vdot(phi0, phi0) 
    #A[0] = np.vdot(phi0,H.dot(phi0))
    VV += np.outer(phi0/norm(phi0), V[0, :])
    for n in range(M):
        #print("n = ", n)
        phi_plus_1 = H.dot(phi0) - A[n]*phi0 - B[n]**2*phi_minus_1
        normphi = norm(phi_plus_1)
        #phi_plus_1 = phi_plus_1/normphi
        ##print("norme de phi n+1 est", normphi)

        if normphi !=0 :
            #print("norme de phi", norm(phi0))
            B[n+1] = normphi / norm(phi0)
            #B[n+1] = np.sqrt(normphi)
            A[n+1] = np.vdot(phi_plus_1, H.dot(phi_plus_1)) / normphi**2 
            #A[n+1] = np.vdot(phi_plus_1, H.dot(phi_plus_1)) 
        else : 
            raise ValueError("the Krylov basis is not independant")
        #print("orthogonality check with n and n-1 ", round(np.abs(np.vdot(phi0,phi_plus_1)),4), round(np.abs(np.vdot(phi_minus_1,phi_plus_1)) ,4) )
        #print("orthogonality check with  n-1 ", np.vdot(phi_minus_1,phi_plus_1) )
        
        #print('norm phi minus one ', norm(phi_minus_1))
        #print(" detail with ortho with n-1 all three, ",np.vdot(phi_minus_1, H.dot(phi0))  ,np.vdot(phi_minus_1, A[n]*phi0),np.vdot(phi_minus_1,B[n]**2*phi_minus_1)  ) 
        #print(" detail first term, ",np.vdot(H.dot(phi_minus_1), phi0)   ) 
        #should_be = np.vdot(phi_plus_1 + A[0]*phi0, H.dot(phi_minus_1)  )
        #print("relation check", should_be)
        phi_minus_1 = phi0.copy()
        phi0 = phi_plus_1.copy()
        #print("orthogonality check with n and n-1 ", round(np.abs(np.vdot(phi_plus_1,phi0)),4), round(np.abs(np.vdot(phi_plus_1,phi_minus_1)) ,4) )
        VV += np.outer(phi0 / norm(phi0), V[n+1, :]) # Because H is in the normalized basis
    B[0] = norm(Psi0) # Useful for the rationalized fraction, if the initial state is chosen the right way
    #B[0] = norm(Psi0/norm(Psi0)) 
    #print("b_0 is ",norm(Psi0) )
    return A, B, VV

def lanczos_AB(Psi0, H, M=100):
    #print("step 6.0.0")
    A, B, _ = lanczos_core(Psi0, H, np.zeros((M+1, 1)), M=M)
    #print("step 6.0.1")
    return A, B

def continued_fraction(w, A, B):
    c = np.zeros_like(w, dtype=np.complex128)
    for n in reversed(range(len(A))):
        c = B[n]**2 / (w - A[n] - c)
    return c

def lanczos_green(w, Psi0, H, M=100, Gamma=0.05, E= 0):
    A, B = lanczos_AB(Psi0, H, M=M)
    A, B = A, B
    A = A -E.real
    return continued_fraction(w + 1j*Gamma, A, B)

def lanczos_gs(Psi0, H, M=100, nev=1):
    A, B = lanczos_AB(Psi0, H, M=M)
    E, V = eigh_tridiagonal(A.real, B[1:].real)
    _, _, VV = lanczos_core(Psi0, H, V[:, :nev], M=M)
    return E[:nev], VV
