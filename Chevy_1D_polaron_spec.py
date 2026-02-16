import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


a=1
#L=30
#n = 0.05
#n= 2/L # for second order Chevy
#N = n*L
#U=0.5
#g = -5
Gamma = 0.05 # broadening
multiple = 1 
cutoff = multiple*2*np.pi # for 3D
#m_I = 1/2
#_B = 1/2  #we assume mass of bosons = mass of impurity
#dk = 2*np.pi/L
#SSB_lambda = -0.001

def chem_potential(U,n,SSB_lambda):
    return -2 + U*n + SSB_lambda/(np.sqrt(n))
    #return U*n           #for continuum 

def free_spec(k,mu):
    return -2*np.cos(k*a) -mu
    #return k**2/(2*m_B) - mu   

def spec_impurity(k):
    return -2*np.cos(k*a ) +2
    #return k**2/(2*m_I) 

def reduced_spec(m1,m2,k,U,n,SSB_lambda):
    mu = chem_potential(U,n,SSB_lambda)
    m_red = m1*m2/(m1 +m2)
    #return k**2/(2*m_red)
    return free_spec(k,mu) + spec_impurity(k) 

def S(k):
    return 4*np.pi*k**2

def inv_g_strength(k_range,t,puls,m1,m2,L,SSB_lambda):
    #polarization_bubble = sum( dk*S(k_range[id])/((2*np.pi)**3)/(puls - reduced_spec(m1,m2,k_range[id],R,U,n,SSB_lambda) -U*n) for id in range(1,len(k_range)) ) #for 3D
    polarization_bubble = 1/L*sum(1/(puls - reduced_spec(m1,m2,k_range[id],U,n,SSB_lambda) +U*n  ) for id in range(1,len(k_range)) )
    return 1/t  + polarization_bubble


def bogo_spec(k,U,n,SSB_lambda):
    Ek = free_spec(k ,chem_potential(U,n,SSB_lambda))
    return np.sqrt(abs((Ek + U*n)*(Ek +3*U*n)))

lambda_list = np.linspace(0,0.5,3)
def Bogo_vs_lambda(lambda_list):
    k_values = np.linspace(-np.pi,np.pi,300)
    for lam in lambda_list:
        plt.plot(k_values, bogo_spec(k_values,U,n,-lam), label =rf'$\lambda$={round(lam,3)}')
    plt.legend(fontsize = 30)
    plt.title(f"U={U}, n={n}",size= 30)
    plt.xlabel("momentum", size =30)
    plt.ylabel("energy", size = 30)
    plt.grid()
    plt.show()
    return None

#Bogo_vs_lambda(lambda_list)
def brightness_vs_lambda(qlist, lambda_list):
    for q in qlist:
        #Ek = free_spec(k ,chem_potential(U,n,SSB_lambda))
        #S = 
        S= [2*n**2*np.pi*(free_spec(q ,chem_potential(U,n,-lambda_el ))+ U*n)/(bogo_spec(q,U,n,-lambda_el)) for lambda_el in lambda_list ]
        plt.plot(lambda_list,S, label =f"q={L*q//(2*np.pi)}" )
    plt.xlabel(r"$\lambda$", size = 30)
    plt.ylabel("DSF spectral brightness", size = 30)
    plt.grid()
    plt.legend(fontsize =30)
    plt.show()
#lambda_list = np.linspace(0,0.5,100)
#qlist = [2*np.pi/L*i for i in range(4) ]
#qlist =[0]
#brightness_vs_lambda(qlist, lambda_list)

def eta(k,U,n,SSB_lambda):
    if k==0 :
        return 0
    else :
        Ek = free_spec(k,chem_potential(U,n,SSB_lambda))
        return 1/4*np.log(abs((Ek + U*n)/(Ek + 3*U*n)))


def uk_eta(k,U,n,SSB_lambda):
    return np.cosh(eta(k,U,n,SSB_lambda))
def vk_eta(k,U,n,SSB_lambda):
    return np.sinh(eta(k,U,n,SSB_lambda))


def Wk(k,U,n,SSB_lambda):
    Ek = free_spec(k,chem_potential(U,n,SSB_lambda))
    epsk = bogo_spec(k,U,n,SSB_lambda)
    uk =uk_eta(k,U,n,SSB_lambda)
    vk =vk_eta(k,U,n,SSB_lambda)
    return uk +vk
    
def V1kkprime(k,k_prime,U,n,SSB_lambda):
    uk =uk_eta(k,U,n,SSB_lambda)
    vk =vk_eta(k,U,n,SSB_lambda)
    uk_prime =uk_eta(k_prime,U,n,SSB_lambda)
    vk_prime =vk_eta(k_prime,U,n,SSB_lambda)
    return uk*uk_prime + vk*vk_prime

#x= np.linspace(0,2,200)
#y = np.array([V1kkprime(2*np.pi/L,2*np.pi/L,x,n,0)])
#y = np.array([Wk(2*np.pi/L,x,n,0)])
#plt.plot(x,y.flatten())
#plt.xlabel("U")
#plt.ylabel(r"$W_{0}$")
#plt.show()

def lippmann_Schwinger_g(k_range,E,U,n,L,SSB_lambda):
    #summation = sum (  dk*S(k_range[id])/((2*np.pi)**3)/(E - free_spec(k_range[id],chem_potential(U,n,SSB_lambda)) - spec_impurity(k_range[id]) -U*n )  for id in range(1,len(k_range)))
    #summation = 1/L*sum (  1/(E - free_spec(k_range[id],chem_potential(U,n,SSB_lambda)) - spec_impurity(k_range[id]) )  for id in range(1,len(k_range)))
    summation = 1/L*sum (  1/(E - reduced_spec(m_I,m_B,k_range[id],U,n,SSB_lambda) - U*n )  for id in range(1,len(k_range)))
    return 1/(summation)


def Frolich_Hamiltonian_new(p,U,n,g,L,SSB_lambda,multiple =1):
    "I organise the psi{p,k} in an array psi_{1,0} psi_{1,1} .... psi_{1,L} psi_{2,0} psi_{2,1} .... psi_{2,L}..... "
    H_Frolich = np.zeros((multiple*L,multiple*L),dtype = np.complex64)
    cste = g/L*sum( vk_eta(2*np.pi/L*k_1,U,n,SSB_lambda)**2  for k_1 in range(1,multiple*L))
    #cste = g*dk/((2*np.pi)**3)*sum(vk_eta(2*np.pi/L*kid,U,n,SSB_lambda)**2*S(2*np.pi/L*kid) for kid in range(1,multiple*L))
    #print('density renormalization',cste)
    for k in range(multiple*L):
        for k_prime in range(multiple*L):
            V = V1kkprime(2*np.pi/L *k ,2*np.pi/L *k_prime,U,n,SSB_lambda) if (k!= 0 and k_prime != 0) else 0
            BI_int_V =  g/L*V 
            impurity_kin =   spec_impurity(  (2*np.pi/L*(L/(2*np.pi)*p-k)) ) if (k==k_prime ) else 0
            bath = bogo_spec(2*np.pi/L*k,U,n,SSB_lambda)  if ( k==k_prime and k!= 0) else 0
            BI_int_cste1 = g*n if ( k==k_prime) else 0
            BI_int_cste2 = cste if (k==k_prime ) else 0
            BI_int_W = 0
            if k == 0 and k_prime != 0:
                BI_int_W = g*np.sqrt(n/L)*Wk(2*np.pi/L*k_prime, U, n,SSB_lambda)
                #BI_int_W = g*np.sqrt(n)*Wk(2*np.pi/L*k_prime,U,n,SSB_lambda)* np.sqrt(S(2*np.pi/L*k_prime)*dk/((2*np.pi)**3)) # for 3D
            elif k != 0 and k_prime == 0:
                BI_int_W= g*np.sqrt(n/L)*Wk(2*np.pi/L*k, U, n, SSB_lambda)
                #BI_int_W = g*np.sqrt(n)*Wk(2*np.pi/L*k,U,n,SSB_lambda)* np.sqrt(S(2*np.pi/L*k)*dk/((2*np.pi)**3))
            #BI_int_W = 0
            #BI_int_V = 0
            #BI_int_cste2 = 0
            #bath = 0
            #impurity_kin = 0
            H_Frolich[ k, k_prime] =  impurity_kin + bath + BI_int_cste1 + BI_int_cste2 + BI_int_W + BI_int_V
    return H_Frolich

def Hamiltonian_2Boglons_approx(p,U,n,g,L,SSB_lambda, multiple=1):
    """I want to organise my psi^p_{k_1,k_2} where the couple  (k_1,k_2) = (0,0), (1,0), (1,1), (2,0), (2,1), (2,2),...(L-1,0),...(L-1,L-1)
    ordered pairs !!
    (k_1,k_2) can be whatever except (0, non zero) :
    a couple (non zero,0) correspond to first order terms
    Remark :the non-primed indices label conjuguated psi and the primed indices label psi"""
    #H_matrice = np.zeros( (  1 + L*(L-1) , 1 + L*(L-1)), dtype = complex)
    H_matrice = np.zeros( (  1 + (L+2)*(L-1)//2 , 1 + (L+2)*(L-1)//2), dtype = complex)
    k1k2 = [(k1,k2) for k1 in range(L) for k2 in range(0,k1+1)]
    print("entered")
    cste = g/L*sum( vk_eta(2*np.pi/L*k_1,U,n,SSB_lambda)**2  for k_1 in range(1,multiple*L)) + g*n
    print('constant',cste)
    #for index in range(1+ L*(L-1)):
    for index in range( 1 + (L+2)*(L-1)//2):
        print('step2')
        #k_1,k_2 = (0,0) if (index ==0) else  ((index-1)//L  +1,  (index-1)%L)
        k_1,k_2 = k1k2[index]
        print('k pairs and index', k_1,k_2, index)
        #for index_prime in range(index, 1+ L*(L-1)):
        for index_prime in range(index, 1 + (L+2)*(L-1)//2):
            #print('step3')
            #k_1_prime,k_2_prime = (0,0) if ( index_prime ==0) else  ((index_prime-1)//L  +1,  (index_prime-1)%L )
            k_1_prime,k_2_prime = k1k2[index_prime]
            same_index = (index == index_prime)
            same_anti_index = (k_1 == k_2_prime and k_2 == k_1_prime) #never satisfied unless index == index_prime
            #print('step4')

            #impurity_kin =   spec_impurity(  2*np.pi/L*(L/(2*np.pi)*p-k_1-k_2)   ) if (index==index_prime ) else 0
            impurity_kin= 0
            if k_2 ==0 and k_2_prime ==0 and same_index: # zeroth order and first order term
                impurity_kin +=   spec_impurity(  2*np.pi/L*(L/(2*np.pi)*p-k_1-k_2)   ) 
            elif k_2 != 0 and k_2_prime !=0: # second order term
                if same_index:
                    impurity_kin += spec_impurity(  2*np.pi/L*(L/(2*np.pi)*p-k_1-k_2)   ) 
                if same_anti_index:
                    impurity_kin += spec_impurity(  2*np.pi/L*(L/(2*np.pi)*p-k_1-k_2)   )  
            #print('step5')
            bath = 0
            if index !=0 and index_prime !=0 and k_2 ==0 and k_2_prime ==0 and k_1 ==k_1_prime: # first order term
                bath += bogo_spec(2*np.pi/L*k_1,U,n,SSB_lambda) 
            elif index !=0 and index_prime !=0 and k_2!=0 and k_2_prime !=0: # second order term
                if same_index:
                    bath += bogo_spec(2*np.pi/L*k_1,U,n,SSB_lambda) + bogo_spec(2*np.pi/L*k_2,U,n,SSB_lambda)
                if same_anti_index:
                    bath+= bogo_spec(2*np.pi/L*k_1,U,n,SSB_lambda) + bogo_spec(2*np.pi/L*k_2,U,n,SSB_lambda)
            #if index != 0 and index_prime != 0 and ( same_index or same_anti_index  ):
            #    bath =  bogo_spec(2*np.pi/L*k_1,U,n,SSB_lambda) if (k_2 == 0 and k_2_prime == 0) else bogo_spec(2*np.pi/L*k_1,U,n,SSB_lambda) + bogo_spec(2*np.pi/L*k_2,U,n,SSB_lambda)

            #BI_cste = cste if (same_index or same_anti_index) else 0
            BI_cste = 0
            if k_2 ==0 and k_2_prime==0 and same_index: # 0 order term and first order
                BI_cste =  cste
            elif k_2!= 0 and k_2_prime != 0 : # second order term
                if same_index:
                    BI_cste += cste
                if same_anti_index:
                    BI_cste += cste

            #BI_V = g/L* V1kkprime(2*np.pi/L*k_1_prime ,2*np.pi/L *k_1,U,n,SSB_lambda) if ((index!= 0 and index_prime != 0) and ( k_2_prime == k_2 or k_2_prime == k_1 or k_1_prime == k_2 or k_1_prime == k_1) ) else 0
            BI_V = 0
            if index != 0 and index_prime!=0 and k_2 ==0 and k_2_prime ==0: #first order term
                BI_V +=  g/L* V1kkprime(2*np.pi/L*k_1_prime ,2*np.pi/L *k_1,U,n,SSB_lambda)
                #BI_V += 0
            elif index!=0 and index_prime!=0 and k_2!=0 and k_2_prime!=0: #second order term
                if k_2 == k_2_prime:
                    BI_V += g/L* V1kkprime(2*np.pi/L*k_1_prime ,2*np.pi/L *k_1,U,n,SSB_lambda)
                if  k_2== k_1_prime:
                    BI_V +=  g/L* V1kkprime(2*np.pi/L*k_2_prime ,2*np.pi/L *k_1,U,n,SSB_lambda)
                if k_1 == k_2_prime :
                    BI_V +=  g/L* V1kkprime(2*np.pi/L*k_1_prime ,2*np.pi/L *k_2,U,n,SSB_lambda)
                if k_1 == k_1_prime :
                    BI_V +=  g/L* V1kkprime(2*np.pi/L*k_2_prime ,2*np.pi/L *k_2,U,n,SSB_lambda)
            BI_W = 0
            if k_2 == k_2_prime and k_2 ==0: # first order term couple psi0 and psi1
                if  k_1 == 0 and k_1_prime != 0:
                    BI_W += g*np.sqrt(n/L)*Wk(2*np.pi/L*k_1_prime, U, n,SSB_lambda) 
                elif k_1 != 0 and k_1_prime == 0:
                    BI_W += g*np.sqrt(n/L)*Wk(2*np.pi/L*k_1, U, n,SSB_lambda)
            elif  k_2 != 0 and k_2_prime == 0 and index_prime !=0: #second order term couple psi1 and psi 2
                if k_2 == k_1_prime :
                    BI_W += g*np.sqrt(n/L)*Wk(2*np.pi/L*k_1, U, n,SSB_lambda) 
                if k_1 == k_1_prime :
                    BI_W += g*np.sqrt(n/L)*Wk(2*np.pi/L*k_2, U, n,SSB_lambda)
            elif k_2 == 0 and k_2_prime != 0 and  index !=0:  #second order term
                if k_1 == k_2_prime:
                    BI_W += g*np.sqrt(n/L)*Wk(2*np.pi/L*k_1_prime, U, n,SSB_lambda)
                if k_1 == k_1_prime :
                    BI_W += g*np.sqrt(n/L)*Wk(2*np.pi/L*k_2_prime, U, n,SSB_lambda)
            #BI_cste =0
            #BI_W =0
            #BI_V =0
            #bath =0
            #impurity_kin =0  
            norm_i = np.sqrt(2.0) if (k_1 == k_2 and k_2!=0) else 1.0
            norm_j = np.sqrt(2.0) if (k_1_prime == k_2_prime and k_2_prime!=0) else 1.0
            H_matrice[index,index_prime] = (BI_cste + BI_W + BI_V +bath + impurity_kin)/(norm_i*norm_j)
            if index != index_prime: #i dont count the diagonal twice
                H_matrice[index_prime,index] = np.conj(H_matrice[index,index_prime] )

    return H_matrice



#M =Hamiltonian_2Boglons_approx(0,U,n,g,L,SSB_lambda)
#print('hermiticity check', np.max(np.abs( M -np.conj(M).T)))
def spectral_function_table(Q_vals, Omega_vals, U, n, g, L, eps, Frolich_Hamiltonian, SSB_lambda):
    t_start = time.time()
    num_Q = len(Q_vals)
    num_Omega = len(Omega_vals)
    eigvals_list = []
    eigvecs_list = []
    for Q in Q_vals:
        print("Polaron Q =", Q)
        H = Frolich_Hamiltonian(Q, U, n, g, L, SSB_lambda)
        print("H polaron finished for that Q")
        eigvals, eigvecs = np.linalg.eigh(H)
        eigvals_list.append(eigvals)
        eigvecs_list.append(eigvecs)
    eigvals_array = np.array(eigvals_list)      
    eigvecs_array = np.stack(eigvecs_list, axis=0)  
    oscillator_strengths = np.abs(eigvecs_array[:, 0, :]) ** 2  
    Omega_vals = Omega_vals[:, None, None] 
    eigvals_array = eigvals_array[None, :, :]  
    osc = oscillator_strengths[None, :, :]   
    denom = Omega_vals - eigvals_array + 1j*eps
    spectral_contributions = osc / denom
    spectral_vals = -2 * np.sum(spectral_contributions.imag, axis=2)  

    print(f"Full spectral function was computed in {time.time() - t_start:.2f} seconds.")
    return spectral_vals
