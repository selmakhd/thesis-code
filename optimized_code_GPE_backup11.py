import numpy as np
import time
import csv
import matplotlib.pyplot as plt
from matplotlib import colors, cm
import pandas as pd
import ast
import cmath
import matplotlib.colors as mcolors
import math
import re
from matplotlib import rc
from matplotlib.pyplot import figure, axes, plot, xlabel, ylabel, title, grid, savefig, show, xticks,yticks
from matplotlib.patches import Patch
from scipy.interpolate import griddata


#################################################################################################################################################
############################################################## Single particle ##################################################################
#################################################################################################################################################

a = 1
#L= 20
p = 1/2
#phi=0
#J_parallel = 1
#J_perp = 0.4
#R= J_perp
#chi =  1*np.pi/2 

#commensurate_chis = [ i/L for i in range(int(L*np.pi))] #rational
#commensurate_chi = commensurate_chis[np.argmin([abs(chi-x) for x in commensurate_chis])]
#print("commensurate chi",commensurate_chi)

#n = 0.5
#n= 1/(2*L)
#N = n*2*L

#t0 = 0.0  
#step = 0.01
#n_steps = 1000
#n_steps = 300000


#R_range = (0.7, 1.5)
#U_range = (0, 0.5)
#R_points = 8
#U_points = 8

#fluc_max = 0
#U=0.2

#all_k_values = [ 2*np.pi/L*k for k in range(-L//2, L//2) ]
spec = lambda R,m,k,chi : -2*np.cos(chi/2)*np.cos(chi/2 -chi*p +k*a)  +(-1)**(m)*np.sqrt( R**2 + 4*np.sin(chi/2)**2*np.sin(chi/2 -chi*p +k*a)**2 ) 
int_MF = lambda R,k,chi,U,omega,n : U*n*(( 3/4*np.sin(2*theta( k,R,chi))**2 -1/2 )*np.sin(2*omega) -1/2*np.sin(2*theta(k,R,chi))**2)





kvals = np.linspace(-np.pi,np.pi,1000)
R, chi, U,n = 5.5, np.pi/2, 0, 0.5
mu = np.min( spec(R, 1, kvals, chi) ) - U*n

q_vals = np.linspace(-np.pi, np.pi, 5)
lowest_eigenvalues = []

for q in q_vals:
    eps_plus =  -2*( np.cos(q + chi/2) -np.cos(chi/2) ) + R -mu
    eps_minus =  -2*( np.cos(q - chi/2) -np.cos(chi/2) ) + R -mu

    M = np.array([
        [eps_plus + U * n,   -U * n,       -R,      0],
        [U * n,              -eps_minus - U * n,  0,       R],
        [-R,                 0,           eps_minus + U * n, -U * n],
        [0,                  R,           U * n,   -eps_plus - U * n]
    ])

    # Compute eigenvalues
    eigvals = np.linalg.eig(M)[0].real  # Sorted real eigenvalues
    lowest_eigenvalues.append(np.min(eigvals))  # Only the lowest (lower branch)

#plt.figure(figsize=(8, 5))
#plt.plot(q_vals, lowest_eigenvalues, label='Lower Branch')
#plt.xlabel(r'$q$')
#plt.ylabel('Lowest Eigenvalue')
#plt.title('Lowest Energy Band vs $q$')
#plt.grid(True)
#plt.legend()
#plt.show()
def victorin_spectrum(k, m, R, chi, U, n,mu):
    # m= + means m=2
    spec2 = spec(R, 2, k, chi) + mu
    spec1 = spec(R, 1, k, chi) + mu

    term1 = (1 / np.sqrt(2)) * (spec2**2 + spec1**2 + 2 * (spec2 + spec1) * U * n + 2 * R**2)

    term2 = (-1)**m * np.sqrt( (spec2**2 - spec1**2)**2 + 4 * U * n * ( spec2**3 + spec1**3 - spec1**2 * spec2 + spec2**2 * spec1) + 4 * R**2 * (     (spec2 + spec1)**2 + 4 * U * n * (spec2 + spec1 + U * n) ))

    return term1 + term2

#kvals = np.linspace(-np.pi,np.pi,1000)
#R, chi, U,n = 5.5, np.pi/2, 0, 0.5
#mu = np.min( spec(R, 1, kvals, chi) )
#yvals1 = victorin_spectrum(kvals,1,R,chi,U,n,mu)
#yvals2 = victorin_spectrum(kvals,2,R,chi,U,n,mu)
#plt.plot(kvals,yvals1)
#plt.plot(kvals,yvals2)
#plt.show()
def plot_spectrum(R_values,chi,L):
    all_k_values = [ 2*np.pi/L*k for k in range(-L//2, L//2) ]
    plt.figure(figsize=(8, 6))
    colors = ['b', 'g', 'r', 'c', 'm']  
    for i, R in enumerate(R_values):
        upper_band = [spec(R, 1, k, chi) for k in all_k_values]
        lower_band = [spec(R, 0, k, chi) for k in all_k_values]
        plt.plot(all_k_values, upper_band, color=colors[i], label=f'R = {R:.2f}',lw=3.5)
        plt.plot(all_k_values, lower_band, color=colors[i],lw=3.5) 
    plt.xlabel("momentum k",size = 30)
    plt.xticks(size=30)
    plt.ylabel(r"$\epsilon_k^\pm$",size = 30)
    plt.yticks(size=30)
    #plt.title(r"Spectra for $\chi=\frac{\pi}{2}$",size = 30)
    #plt.title(r"$\chi=\frac{\pi}{2}$",size = 30)
    plt.title(r"$\chi=0.5\pi$",size = 30)
    #plt.legend(fontsize = 30)
    plt.legend(loc='upper right', fontsize=30, frameon=True)
    plt.grid()
    plt.ylim(-3.5,3.5)
    plt.show()

R_values = np.linspace(0, 1.5, 5)
R_values = [2.0]
#plot_spectrum(R_values,0.5*np.pi,100)

def plot_spectrum_vs_chi(chi_values,R,L):
    all_k_values = [ 2*np.pi/L*k for k in range(-L//2, L//2) ]
    plt.figure(figsize=(8, 6))
    colors = ['b', 'g', 'r', 'c', 'm']  
    for i, chi in enumerate(chi_values):
        upper_band = [spec(R, 1, k, chi) for k in all_k_values]
        lower_band = [spec(R, 0, k, chi) for k in all_k_values]
        plt.plot(all_k_values, upper_band, color=colors[i], label= r'$\chi$'f'= {chi:.2f}',lw=3.5)
        plt.plot(all_k_values, lower_band, color=colors[i],lw=3.5) 
    plt.xlabel("momentum k",size = 30)
    plt.xticks(size=30)
    plt.ylabel(r"$\epsilon_k^\pm$ ",size = 30)
    plt.yticks(size=30)
    plt.title(f"Spectra for R={R}",size = 30)
    plt.legend(fontsize = 20)
    plt.grid()
    plt.ylim(-3.5,3.5)
    plt.show()

chi_values = np.linspace(0, np.pi, 5)
#plot_spectrum_vs_chi(chi_values,1,100)

   
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

def plot_arcsin_gamma_color(R_values, chi_values, chi_points=None, R_fixed=0.5):
    arcsin_gamma = np.zeros((len(R_values), len(chi_values)))

    for j, chi in enumerate(chi_values):
        for i, R in enumerate(R_values):
            arcsin_gamma[i, j] = np.arcsin(gamma(R, chi)) / np.pi

    chi_grid, R_grid = np.meshgrid(chi_values, R_values)
    plt.figure(figsize=(8, 6))
    c = plt.pcolormesh(chi_grid, R_grid, arcsin_gamma, shading='auto', cmap='viridis')

    # Colorbar
    cb = plt.colorbar(c)
    cb.set_label(r'Vortex density', fontsize=35, labelpad=30)
    cb.ax.tick_params(labelsize=18)

    # Red theoretical boundary line
    chi_fine = np.linspace(0.01, 1.8, 1000)
    R_theory = 2 * np.tan(chi_fine / 2) * np.sin(chi_fine / 2)
    plt.plot(chi_fine, R_theory, color='red', lw=3, label=r'$R_{th}=2\tan(\frac{\chi}{2})\sin(\frac{\chi}{2})$')

    # Pretty white stars with black edges
    if chi_points is not None:
        R_stars = [R_fixed] * len(chi_points)
        plt.scatter(chi_points, R_stars, marker='*', s=200, color='white', edgecolor='black', zorder=5)

    # Labels and styling
    plt.xlabel(r'$\chi$', fontsize=35)
    plt.ylabel('R', fontsize=35, rotation=0, labelpad=20)
    #plt.title(r'Vortex density vs $R$ and $\chi$', fontsize=35, pad=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.2)
    plt.legend(fontsize=25, loc='upper right')
    plt.tight_layout()
    plt.show()

    # Cuts for fixed R
    R_index = np.abs(R_values - R_fixed).argmin()
    plt.figure(figsize=(7, 4))
    plt.plot(chi_values, arcsin_gamma[R_index, :], lw=2)
    plt.title(fr'Cut at $R = {R_values[R_index]:.3f}$', fontsize=24)
    plt.xlabel(r'$\chi$', fontsize=20)
    plt.ylabel(r'Vortex density', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

    # Cuts for fixed chi
    chi_target = np.pi / 2
    chi_index = np.abs(chi_values - chi_target).argmin()
    plt.figure(figsize=(7, 4))
    plt.plot(R_values, arcsin_gamma[:, chi_index], lw=2)
    plt.title(fr'Cut at $\chi = {chi_values[chi_index]:.3f}$', fontsize=24)
    plt.xlabel('R', fontsize=20)
    plt.ylabel(r'Vortex density', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

#Example usage
#R_vals = np.linspace(0, 2, 1000)
#chi_vals = np.linspace(0, np.pi, 1000)
#plot_arcsin_gamma_color(R_vals, chi_vals, chi_points=[np.pi*0.99, np.pi*0.4, np.pi*0.35, np.pi*0.2], R_fixed=0.5)

def plot_theta(R_values,chi,L):
    all_k_values = [ 2*np.pi/L*k for k in range(-L//2, L//2) ]
    plt.figure(figsize=(8, 6))
    colors = ['b', 'g', 'r', 'c', 'm']  
    for i, R in enumerate(R_values):
        theta_vals = [theta(k,R,chi) for k in all_k_values]
        plt.plot(all_k_values, theta_vals, color=colors[i], label=f'R = {R:.2f}',lw=3.5)
    plt.xlabel("momentum k",size = 30)
    plt.xticks(size=30)
    plt.ylabel(r"$\theta_k$ ",size = 30)
    plt.yticks(size=30)
    plt.title(r" $\chi=\frac{\pi}{2}$",size = 30)
    plt.legend(fontsize = 20)
    plt.grid()
    plt.show()

R_values = np.linspace(0.0000000000001, 1.5, 5)
#plot_theta(R_values,np.pi/2,100)

def plot_theta_vs_chi(chi_values,R,L):
    all_k_values = [ 2*np.pi/L*k for k in range(-L//2, L//2) ]
    plt.figure(figsize=(8, 6))
    colors = ['b', 'g', 'r', 'c', 'm']  
    for i, chi in enumerate(chi_values):
        theta_vals = [theta(k,R,chi) for k in all_k_values]
        plt.plot(all_k_values, theta_vals, color=colors[i], label= r'$\chi$'+f'= {chi:.2f}',lw=3.5)
    plt.xlabel("momentum k",size = 30)
    plt.xticks(size=30)
    plt.ylabel(r"$\theta_k$ ",size = 30)
    plt.yticks(size=30)
    plt.title(f" R={R}",size = 30)
    plt.legend(fontsize = 20)
    plt.grid()
    plt.show()

chi_values = np.linspace(0.0, np.pi, 5)
#plot_theta_vs_chi(chi_values,1,100)

def k_minus(R,chi,L): #only for the symmetric gauge
    all_k_values = [ 2*np.pi/L*k for k in range(-L//2, L//2) ]
    k_minus_analytical = - np.arcsin(gamma(R,chi))/a
    delta = [abs(k-k_minus_analytical) for k in all_k_values]
    i=np.argmin(delta)
    return all_k_values[i]
def k_plus(R,chi,L):
    all_k_values = [ 2*np.pi/L*k for k in range(-L//2, L//2) ]
    k_plus_analytical = + np.arcsin(gamma(R,chi))/a
    delta = [abs(k-k_plus_analytical) for k in all_k_values]
    i=np.argmin(delta)
    return all_k_values[i]
    #return np.arcsin(gamma(R,chi))


def plot_kmin_vs_param(R_values, chi_values):
    def kmin_vs_R(R_values, chi_fixed):
        kmin_vals = []
        for R in R_values:
            g = gamma(R, chi_fixed)
            val = np.arcsin(g)
            kmin_vals.append([val, -val])
        return np.array(kmin_vals)
    def kmin_vs_chi(chi_values, R_fixed):
        kmin_vals = []
        for chi in chi_values:
            g = gamma(R_fixed, chi)
            val = np.arcsin(g)
            kmin_vals.append([val, -val])
        return np.array(kmin_vals)
    kmins_R = kmin_vs_R(R_values, chi_fixed=np.pi/2)
    plt.figure(figsize=(7, 5))
    plt.plot(R_values, kmins_R[:, 0],  color='blue',lw =5)
    plt.plot(R_values, kmins_R[:, 1], color='blue',lw =5)
    plt.xlabel('R', fontsize=35)
    plt.ylabel(r'$k_{\min}$', fontsize=35)
    plt.title(r'$k_{\min}$ vs $R$ at $\chi = \frac{\pi}{2}$', fontsize=35)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.show()
    kmins_chi = kmin_vs_chi(chi_values, R_fixed=1)
    plt.figure(figsize=(7, 5))
    plt.plot(chi_values, kmins_chi[:, 0], color='blue',lw=5)
    plt.plot(chi_values, kmins_chi[:, 1], color='blue',lw =5)
    plt.xlabel(r'$\chi$', fontsize=35)
    plt.ylabel(r'$k_{\min}$', fontsize=35)
    plt.title(r'$k_{\min}$ vs $\chi$ at $R = 1$', fontsize=35)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.show()
R_values = np.linspace(0, 2, 500)
chi_values = np.linspace(0, np.pi, 500)

#plot_kmin_vs_param(R_values, chi_values)
def MF_kmin(k_vals, U, n, chi, R):
    k_above = []
    k_below = []
    for k in k_vals:
        A = (3/4*np.sin(2*theta(k, R, chi))**2 - 1/2)    
        if A > 0:
            k_above.append(k)
        if A <= 0:
            k_below.append(k)
    omega_0_sector = [spec(R, 1, k, chi) + int_MF(R, k, chi, U, 0, n) for k in k_above]
    omega_pi4_sector = [spec(R, 1, k, chi) + int_MF(R, k, chi, U, np.pi/4, n) for k in k_below]
    if len(omega_pi4_sector) == 0:
        idx = int(np.argmin(omega_0_sector))
        k_min = k_above[idx]
        omega_min = 0
    elif len(omega_0_sector) == 0:
        idx = int(np.argmin(omega_pi4_sector))
        k_min = k_below[idx]
        omega_min = np.pi/4
    else:
        idx_0 = int(np.argmin(omega_0_sector))
        idx_pi4 = int(np.argmin(omega_pi4_sector))
        if omega_pi4_sector[idx_pi4] < omega_0_sector[idx_0]:
            k_min = k_below[idx_pi4]
            omega_min = np.pi/4
        else:
            k_min = k_above[idx_0]
            omega_min = 0
    return (-abs(k_min), abs(k_min), omega_min)

#print(MF_kmin(all_k_values,U,n,chi,R))
def kmin_vs_R_phases(k_vals, R_vals, U, chi,n):
    """plot kmin vs R but colors different phases in different colors Green = vortex, Red = meissner, purple = Biased"""
    minima1 = []
    colors = []
    for R in R_vals:
        k_above = []
        k_below = []
        for k in k_vals:
            A = (3/4 * np.sin(2 * theta( k, R, chi))**2 - 1/2)
            if A > 0:
                k_above.append(k)
            else:
                k_below.append(k)
        omega_0_sector = [spec(R, 1, k, chi) + int_MF(R, k, chi, U, 0, n) for k in k_above]
        omega_pi4_sector = [spec(R, 1,  k, chi) + int_MF(R, k, chi, U, np.pi/4, n) for k in k_below]
        min_k_0 = k_above[np.argmin(omega_0_sector)] if omega_0_sector else None
        min_k_pi4 = k_below[np.argmin(omega_pi4_sector)] if omega_pi4_sector else None
        if min_k_0 is None:
            min_k = min_k_pi4
            phase_color = "green"
        elif min_k_pi4 is None:
            min_k = min_k_0
            phase_color = "purple"  
        else:
            if omega_pi4_sector[np.argmin(omega_pi4_sector)] < omega_0_sector[np.argmin(omega_0_sector)]:
                min_k = min_k_pi4
                phase_color = "green"  
            else:
                min_k = min_k_0
                phase_color = "purple" 
        if abs(min_k) < 1e-3: 
            phase_color = "red"
        minima1.append(min_k)
        colors.append(phase_color)
    minima1 = minima1 + [-k for k in minima1]
    colors = colors + colors 
    R_vals = list(R_vals) + list(R_vals)  
    plt.scatter(R_vals, minima1, c=colors, s=25, label=f'χ = {chi}, U = {U}')
    plt.title(fr' χ={round(chi/np.pi,2)}$\pi$, U={U}, n = {n}',fontsize = 30)
    plt.xlabel(r'$R$', fontsize=30)
    plt.ylabel(r'$k_{min}$', fontsize=30)
    plt.grid()
    plt.show()
chi= 0.5*np.pi
U= 0.175
k_vals = np.linspace(0, np.pi, 500)  
R_vals = np.linspace(0, 3, 300)  
#kmin_vs_R_phases(k_vals, R_vals, U, chi,0.5)

def kmin_vs_chi_phases(k_vals,chi_vals,U,R,n):
    """plot kmin vs chi but colors different phases in different colors Green = vortex, Red = meissner, purple = Biased"""
    minima1 = []
    colors = [] 
    for chi in chi_vals:
        k_above = []
        k_below = []
        for k in k_vals:
            A = (3/4 * np.sin(2 * theta( k, R, chi))**2 - 1/2)
            if A > 0:
                k_above.append(k)
            else:
                k_below.append(k)
        omega_0_sector = [spec(R, 1,  k, chi) + int_MF(R, k, chi, U, 0, n) for k in k_above]
        omega_pi4_sector = [spec(R, 1, k, chi) + int_MF(R, k, chi, U, np.pi/4, n) for k in k_below]
        min_k_0 = k_above[np.argmin(omega_0_sector)] if omega_0_sector else None
        min_k_pi4 = k_below[np.argmin(omega_pi4_sector)] if omega_pi4_sector else None
        if min_k_0 is None:
            min_k = min_k_pi4
            phase_color = "green"  
        elif min_k_pi4 is None:
            min_k = min_k_0
            phase_color = "purple" 
        else:
            if omega_pi4_sector[np.argmin(omega_pi4_sector)] < omega_0_sector[np.argmin(omega_0_sector)]:
                min_k = min_k_pi4
                phase_color = "green" 
            else:
                min_k = min_k_0
                phase_color = "purple"  
        if abs(min_k) < 1e-3: 
            phase_color = "red"
        minima1.append(min_k)
        colors.append(phase_color)
    minima1 = minima1 + [-k for k in minima1]
    colors = colors + colors  
    chi_vals = list(chi_vals) + [x for x in chi_vals] 
    plt.scatter(chi_vals, minima1, c=colors, s=25, label=f'U = {U}')
    plt.title(f'R={R},U= {U},n={n}',fontsize = 30)
    plt.xlabel(r'$\chi$', fontsize=30)
    plt.ylabel(r'$k_{min}$', fontsize=30)
    plt.grid()
    plt.show()
#R = 1
#U= 0.175
#chi_vals = np.linspace(0, np.pi, 200)
#kmin_vs_chi_phases(k_vals,chi_vals,U,R,0.5)
def get_minimizing_k_and_phase(k_vals, chi, R, U, n):
    """Find the minimizing k and determine the phase type for a given chi, R, U, n.
    Now using MF_kmin. """
    k_min_minus, k_min_plus, omega_min = MF_kmin(k_vals, U, n, chi, R)
    chosen_k = k_min_plus
    if abs(chosen_k) < 0.01:
        chosen_phase = 'Meissner'
    else:
        if np.isclose(omega_min, 0):
            chosen_phase = 'Biased Ladder'
        else:
            chosen_phase = 'Vortex'
    return chosen_k, chosen_phase

phase_colors_list = ['r', 'g', 'b'] 

def create_phase_diagram(k_vals, R_vals, chi_vals, U, n):
    """
    Create a phase diagram by minimizing with respect to k for each pair of (R, chi).
    Color the phases as red (Meissner), green (Vortex), and white (Biased Ladder).
    """
    phase_to_num = {'Meissner': 0, 'Biased Ladder': 1, 'Vortex': 2 }
    phase_grid = np.zeros((len(R_vals), len(chi_vals)), dtype=int)
    for i, R in enumerate(R_vals):
        for j, chi in enumerate(chi_vals):
            _, phase = get_minimizing_k_and_phase(k_vals, chi, R, U, n)
            print('diag',phase_to_num[phase],'R,chi=',R,chi)
            phase_grid[i, j] = phase_to_num[phase] 
        end_time = time.time()
        print(start_time - end_time)
    plt.figure(figsize=(10, 6))
    cmap = colors.ListedColormap(['red', 'white', 'green']) 
    bounds = [-0.1, 0.9, 1.9, 2.9] 
    norm = colors.BoundaryNorm(bounds, cmap.N)
    print("Phase Grid Shape:", phase_grid.shape)
    print("Phase Grid Values:\n", phase_grid)
    plt.imshow(phase_grid, extent=[min(chi_vals), max(chi_vals), min(R_vals), max(R_vals)],  origin='lower', aspect='auto', cmap=cmap, norm=norm)
    plt.draw() 
    plt.xlabel(r'$\chi$', fontsize=30)
    plt.xticks(fontsize = 30)
    plt.ylabel(r'$R$', fontsize=30)
    plt.yticks(fontsize = 30)
    plt.title(f'U = {U}', fontsize=30)
    handles = [ Patch(facecolor='red', edgecolor='black', label='Meissner'), Patch(facecolor='white', edgecolor='black', label='Biased Ladder'),Patch(facecolor='green', edgecolor='black', label='Vortex') ]
    plt.legend(handles=handles, title="Phase", loc="best",fontsize=30)
    plt.show()
start_time = time.time()
k_vals = np.linspace(0,np.pi,100)
R_vals =  np.linspace(0,2,20)
chi_vals =  np.linspace(0,np.pi,20)
U, n = 0.005,0.5
#create_phase_diagram(k_vals, R_vals, chi_vals, U, n)

def create_phase_diagram_R_vs_U(k_vals, R_vals, U_vals, chi_fixed, n):
    """
    Create a phase diagram for R vs U with fixed chi.
    """
    phase_to_num = {'Meissner': 0,'Biased Ladder': 1,'Vortex': 2}
    phase_grid = np.zeros((len(R_vals), len(U_vals)), dtype=int)
    for i, R in enumerate(R_vals):
        for j, U in enumerate(U_vals):    
            _, phase = get_minimizing_k_and_phase(k_vals, chi_fixed, R, U, n)
            phase_grid[i, j] = phase_to_num[phase] 
        end_time = time.time()
        print(start_time - end_time)
    plt.figure(figsize=(10, 6))
    cmap = colors.ListedColormap(['red', 'white', 'green']) 
    bounds = [-0.1, 0.9, 1.9, 2.9] 
    norm = colors.BoundaryNorm(bounds, cmap.N)
    plt.imshow(phase_grid, extent=[min(U_vals), max(U_vals), min(R_vals), max(R_vals)], origin='lower', aspect='auto', cmap=cmap, norm=norm)
    plt.xlabel(r'$U$', fontsize=30)
    plt.xticks(fontsize=30)
    plt.ylabel(r'$R$', fontsize=30)
    plt.yticks(fontsize=30)
    plt.title(f'$\\chi = {round(chi_fixed/np.pi,2)}\pi$', fontsize=30)
    handles = [ Patch(facecolor='red', edgecolor='black', label='Meissner'), Patch(facecolor='white', edgecolor='black', label='Biased Ladder'), Patch(facecolor='green', edgecolor='black', label='Vortex')  ]
    plt.legend(handles=handles, title="Phase", loc="best", fontsize=20)
    plt.show()

start_time = time.time()

#create_phase_diagram_R_vs_U(k_vals, R_vals, U_vals, chi_fixed, n)

def chem_pot(R,chi):
    return U/2*(2*n-1) + spec(R,1,-chi/2,chi)


def Euler_explicit(f,t0,y0,step,n_steps):
    t_values = [t0]
    y_values = [y0]
    for i in range(1,n_steps):
        t_values.append(t_values[-1] + step)
        y_values.append( y_values[-1] + step* f(y_values[-1],t_values[-1]))
    return t_values,y_values

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
    psi = y.reshape((L, 2))
    dydt = np.zeros_like(psi, dtype=complex)
    psi_left = np.roll(psi, shift=1, axis=0)
    psi_right = np.roll(psi, shift=-1, axis=0)
    dydt[:, 0] = +J_parallel*(np.exp(-1j*chi*(1-p))*psi_left[:, 0] + np.exp(1j*chi*(1-p))*psi_right[:, 0]) + J_perp*psi[:, 1] - (U/2)*(2*abs(psi[:, 0])**2 ) * psi[:, 0]
    dydt[:, 1] = +J_parallel*(np.exp(1j*chi*p)*psi_left[:, 1]+np.exp(-1j*chi*p)*psi_right[:, 1]) + J_perp * psi[:, 0] - (U / 2)*(2*abs(psi[:, 1])**2 )*psi[:, 1]
    return dydt.flatten() 


#removed cst constribution of interaction 
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
    interaction = (U/2)*(np.sum(abs(psis[:, 0])**4) *2 -  abs(psis[:, 0])@abs(psis[:, 0]) + np.sum(abs(psis[:, 1])**4) *2 -  abs(psis[:, 1])@abs(psis[:, 1]))
    summation = kin1 + kin2 + kin3 + interaction
    return summation.real

#removed cst constribution of interaction  
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

def psi_mix_k(k,j,m,R,chi,L,omega,phi): #for different k in symetric gauge
    if k ==0:
        omega=0
    real1 = 1/np.sqrt(L)*np.cos( theta(k,R,chi) +np.pi/2*(2-m))*np.cos(omega)*np.cos(j*k*a)
    real2 = 1/np.sqrt(L)*np.cos(theta(-k,R,chi) +np.pi/2*(2-m))* np.sin(omega)*np.cos(-j*k*a + phi)
    imaginary1 = 1/np.sqrt(L)*np.cos(theta(k,R,chi) +np.pi/2*(2-m))*(np.cos(omega)*np.sin(j*k*a) )
    imaginary2 = 1/np.sqrt(L)*np.cos(theta(-k,R,chi) +np.pi/2*(2-m))*(np.sin(omega)*np.sin(-j*k*a + phi))
    return real1 +real2 +1j*(imaginary1 +imaginary2)
def psi_k(k,R,chi,L):
    psi = np.zeros(2*L,dtype = complex)
    for j in range(L):
        psi[2*j] =np.exp(1j*j*k)*np.cos( theta(k,R,chi) +np.pi/2*(2-1))/np.sqrt(L)
        psi[2*j +1] = np.exp(1j*j*k)*np.cos( theta(k,R,chi) +np.pi/2*(2-2))/np.sqrt(L)
    return psi
def psi_mix_k_new(k,R,chi,L,omega,phi):
    if k ==0:
        omega =0
    return np.cos(omega)* psi_k(k,R,chi,L) + np.sin(omega)*np.exp(1j*phi)* psi_k(-k,R,chi,L)


def density_analytical(k,j,m,R,chi,L,omega,phi):
    homogeneous_plus = np.cos(omega)**2*np.cos(theta(k,R,chi) + np.pi/2*(2-m))**2/L 
    homogeneous_minus = np.sin(omega)**2*np.cos(theta(- k,R,chi) + np.pi/2*(2-m))**2/L
    oscillating_term = homogeneous_minus + homogeneous_plus -np.sin(2*omega)*np.sin(2*theta(k,R,chi))/(2*L)*np.cos(2*j*k*a - phi)
    return   oscillating_term
######################################################################################################################################################################
############################################################### Beating effect #######################################################################################
######################################################################################################################################################################

#L = 100
#x = np.linspace(1, L, L)  
#zeros = np.zeros_like(x)
#xbis = np.linspace(1, L, 10000)  
#q = 3
#ep = 0.02
#k = np.pi / q * (1 + ep)
#y = np.cos(2 * x * k)
#k1 = np.pi / q
#k2 = k  
#y_modulated = np.cos(xbis * (k1 + k2)) * np.cos(xbis * (k1 - k2))  
#plt.figure(figsize=(10, 5))
#plt.plot(x, y, lw=4, label="Discrete lattice: $k=\\pi/q(1 + \epsilon)$")
#plt.plot(xbis, y_modulated, color='r', lw=2, label="Beating: superposition of $k_1, k_2$")
#plt.legend()
#plt.grid(True)
#plt.show()



#L=100
#x = np.linspace(1,L,L) # L=1
#zeros = np.zeros_like(x)
#xbis = np.linspace(1,L,10000)
#z = [0, 0.25, 0.5, 0.75, 1]
#zy = [0,0,0,0,0]
#q=3
#ep = 0.02
#k = np.pi/q*(1+ep)
#y = np.cos(2*x*k)
#ybis1 = np.cos(xbis*1/q*np.pi)*np.cos(xbis*1/q*np.pi*ep)  
#ybis2 = np.sin(xbis*1/q*np.pi)*np.sin(xbis*1/q*np.pi*ep)
#ybis =np.cos(xbis*k)
#plt.plot(x,y,lw =4)
#plt.plot(xbis,ybis, lw =1)
#plt.plot(xbis,ybis1, color ='r',lw =1)
#plt.plot(xbis,ybis2, color ='g',lw=1)
#plt.scatter(z,zy)
#plt.scatter(x,zeros)
#plt.show()
#L=200
#L=300
#omega= np.pi/4
#x = np.arange(1,L+1)

#k= 2*np.pi/200*49/2
#k = 0.17*np.pi
#q= 50/10
#k= np.pi/(q)*0.96
#ep =2*k - np.pi/3
#k = 1/3*np.pi/2
#k = k_plus(R,chi,L)
#print('k plus is ', k*L/(2*np.pi))
#print('first ans second and third ',density_analytical(k,0,1,R,chi,L,omega,phi),density_analytical(k,1,1,R,chi,L,omega,phi),density_analytical(k,2,1,R,chi,L,omega,phi))
#m=1
#psi_v = [abs(psi_mix_k(k,j,1,R,chi,L,omega,phi))**2 for j in range(L)]
#psi_v = [density_analytical(k,j,1,R,chi,L,omega,phi) for j in range(L)]
#psi_vbis = [np.cos(omega)**2*np.cos(theta(k,R,chi) + np.pi/2*(2-m))**2/L +np.sin(omega)**2*np.cos(theta(- k,R,chi) + np.pi/2*(2-m))**2/L-np.sin(2*omega)*np.sin(2*theta(k,R,chi))/(2*L)*np.cos(1*el*(ep)*a - phi) for el in xbis]
#psi_vbisbis = [np.cos(omega)**2*np.cos(theta(k,R,chi) + np.pi/2*(2-m))**2/L +np.sin(omega)**2*np.cos(theta(- k,R,chi) + np.pi/2*(2-m))**2/L-np.sin(2*omega)*np.sin(2*theta(k,R,chi))/(2*L)*np.cos(1*(el-np.pi/(2*ep))*(ep)*a - phi) for el in xbis]
#print(psi_v)
#plt.plot(x,psi_v,linestyle = '-')
#plt.xticks([i for i in range(1,L+1)])
#plt.plot(xbis,psi_vbis,linestyle = '-',color ='r')
#plt.plot(xbis,psi_vbisbis,linestyle = '-',color ='g')
#plt.show()

######################################################################################################################################################################
############################################################## Gross PItaevskii Evolution  ###########################################################################
######################################################################################################################################################################


def density_imbalance_y(psis,L):
    psis = psis.reshape(L,2)
    leg1_minus_leg2 = abs(psis[:, 0])@abs(psis[:, 0]) - abs(psis[:, 1])@abs(psis[:, 1])
    leg1_plus_leg2 = abs(psis[:, 0])@abs(psis[:, 0]) + abs(psis[:, 1])@abs(psis[:, 1])
    return abs(leg1_minus_leg2)/leg1_plus_leg2

def vortex_density(psis,L):
    psis_real = [abs(psi) for psi in psis]
    mean = sum(psis_real)/len(psis_real)
    psis = psis.reshape(L,2)
    modulus = abs(psis[0,0])
    vortex_number = 0
    for j in range(1,L):
        if (modulus-mean)*(abs(psis[j,0])-mean) < -0.00000001:
            vortex_number += 1
            modulus = abs(psis[j,0])
    vortex_density = vortex_number//2/L
    return vortex_density
#ivan way
def vortex_density_new(psis,L):
    summation = 0
    for j in range(L):
        delta= np.angle(np.conj(psis[2*((j+1)%L)])*psis[2*((j+1)%L)+1]/(np.conj(psis[2*j])*psis[2*j+1]))
        summation += delta
    vortex_density = summation/(2*np.pi)/L
    return np.abs(vortex_density)

def vortex_density_k(k,L):
    return k/np.pi # because vortex size = M = np.pi/k and the number of vortices is L/M and the vortex density 1/M= k/np.pi
#L= 1000
#p = 1/2
#a=1
#omega = np.pi/4
#chi = np.pi/2
#R = 0.01
#k =  k_minus(R,chi,L)
#phi = 0
#psi_test = np.zeros(2*L, dtype = complex )
#for i in range(L):
#    psi_test[2*i] = psi_mix_k(k,i,1,R,chi,L,omega,phi)
#    psi_test[2*i + 1] = psi_mix_k(k,i,2,R,chi,L,omega,phi)
#print("vortex density", vortex_density_new(psi_test,L), vortex_density(psi_test,L))


def perp_current(omega,phi,j,R,chi,L):
    if R<R_threshold(chi):
        return -R/(L)*np.sin(2*omega)*np.cos(2*theta(k_plus(R,chi,L),R,chi))*np.sin(j*2*k_plus(R,chi,L) - phi) 
    else:
        return 0

def perp_current_normalized(omega,phi,j,R,chi,L):
    perp_curr = [ perp_current(omega,phi,j_val,R,chi,L) for j_val in range(L)]
    norm = np.max(np.abs(perp_curr ))
    if R<R_threshold(chi):
        #norm = np.abs(R/(L)*np.sin(2*omega)*np.cos(2*theta_plus(gamma(R,chi), R,chi)))
        return perp_current(omega,phi,j,R,chi,L)/norm
    else:
        return 0

def perp_current_functional(psi,j,m,R,L): #outgoing
    psis = np.reshape(psi, (L,2))
    part1 = -1j*R*np.conj(psis[j,m])*psis[j,1-m]
    return part1 + np.conj(part1)

def parallel_current(omega,phi,j,m,R,chi,L):
    if R< R_threshold(chi):
        a1 = 1/(L)*np.sin((-1)**(m-1)*chi/2 + k_plus(R,chi,L))*np.cos(omega)**2*(1 + (-1)**m* np.cos(2*theta(k_plus(R,chi,L),R,chi))) 
        a2 = 1/(L)*np.sin((-1)**(m-1)*chi/2 - k_plus(R,chi,L))*np.sin(omega)**2*(1 + (-1)**(m-1)* np.cos(2*theta(k_plus(R,chi,L),R,chi))) 
        a3 =  (-1)**(m+1)*1/(L)*np.sin(chi/2)*np.sin(2*omega)*np.sin(2*theta(k_plus(R,chi,L),R,chi))
        a1,a2 = 0,0
        return a1 + a2 + a3
    else:
        return (-1)**(m+1)*np.sin(chi/2)

def parallel_current_functional(psi,j,m,R,chi,L):
    psis = np.reshape(psi, (L,2))
    #part1 = -1j*np.exp(1j*chi*(1-p)*(1/2-m))*np.conj(psis[j,m])*psis[(j+1)%(L-1),m]
    part1 = 1j*np.exp(1j*chi*(1-p)*(1-2*m))*np.conj(psis[(j+1)%(L-1),m])* psis[j,m]
    return part1 + np.conj(part1)

def chiral_current_functional(psis,J_parallel,chi,L):
    psis = psis.reshape(L,2)
    psis_right = np.roll(psis, shift=-1, axis=0)
    current = -1j*J_parallel*(np.exp(1j*chi*(1-p))*psis[:, 0].conj()@psis_right[:, 0] - np.exp(-1j*chi*(p))*psis[:, 1].conj()@psis_right[:, 1] - np.exp(-1j*chi*(1-p))*psis[:, 0]@psis_right[:, 0].conj() + np.exp(1j*chi*(p))*psis[:, 1] @psis_right[:, 1].conj())
    return 1/L*current.real
def chiral_current(k,R,chi,L):
    if k>0.0001:
        return (1-np.cos(2*theta(k,R,chi)))/L*np.sin(chi/2+k) + (1-np.cos(2*theta(k,R,chi)))/L*np.sin(chi/2+k)
    return 2/L*np.sin(chi/2)
def plot_chiral_current_vs_R(chi=np.pi/2, L=1):
    R_vals = np.linspace(0.001, 3, 500)
    y_vals = [chiral_current(gamma(R, chi), R, chi, L) for R in R_vals]
    max_val = np.max(np.abs(y_vals))
    plt.plot(R_vals, y_vals/max_val,lw = 5)
    plt.xlabel("R",size =30)
    plt.ylabel(r"$\frac{\langle \hat{J}_c \rangle}{J_c^0}$",size =30)
    plt.title(r"$\chi = \frac{\pi}{2}$",size =30)
    plt.grid(True)
    plt.show()

def plot_chiral_current_vs_chi(R=1, L=1):
    chi_vals = np.linspace(0, np.pi, 500)
    y_vals = [chiral_current(gamma(R, chi), R, chi, L) for chi in chi_vals]
    max_val = np.max(np.abs(y_vals))
    plt.plot(chi_vals, y_vals/max_val,lw = 5)
    plt.xlabel(r"$\chi$",size =30)
    plt.ylabel(r"$\frac{\langle \hat{J}_c \rangle}{J_c^0}$",size =30)
    plt.title(f"R ={R}",size =30)
    plt.grid(True)
    plt.show()

#plot_chiral_current_vs_R()
#plot_chiral_current_vs_chi()

def average_parallel_current_functional_leg1(psis,J_parallel,L):
    psis = psis.reshape(L,2)
    psis_right = np.roll(psis, shift=-1, axis=0)
    current = -1j * J_parallel * ( np.exp(1j*chi*(p))*psis[:, 0].conj()@psis_right[:, 0]  - np.exp(-1j*chi*(p))*psis[:, 0]@psis_right[:, 0].conj() )
    return 1/L*current.real
def average_parallel_current_functional_leg2(psis,J_parallel,L):
    psis = psis.reshape(L,2)
    psis_right = np.roll(psis, shift=-1, axis=0)
    current = -1j * J_parallel * ( np.exp(-1j*chi*(1-p))*psis[:, 1].conj() @ psis_right[:, 1]  - np.exp(1j*chi*(1-p))*psis[:, 1]@psis_right[:, 1].conj() )
    return 1/L*current.real
def parallel_current(omega,phi,j,m,R,chi,L):
    if R < R_threshold(chi):
        a1 = -1/(L)*np.sin((-1)**(m-1)*chi/2 + k_plus(R,chi,L)) *np.cos(omega)**2*(1 + (-1)**m* np.cos(2*theta(k_plus(R,chi,L),R,chi)))
        #a1 = 0
        a2 = -1/(L)*np.sin((-1)**(m-1)*chi/2 - k_plus(R,chi,L) ) *np.sin(omega)**2*(1 + (-1)**(m-1)* np.cos(2*theta(k_plus(R,chi,L),R,chi))) 
        #a2 = 0
        a3 = (-1)**(m+1)*1/(L)*np.sin(chi/2)*np.sin(2*omega)*np.sin(2*theta(k_plus(R,chi,L),R,chi))*np.cos(2*j*k_plus(R,chi,L) + k_plus(R,chi,L) -phi)
        a1,a2 = 0,0
        return a1 + a2 + a3
    else :
        return 1/L*(-1)**(m+1)*np.sin(chi/2)

def parallel_current_normalized(omega,phi,j,m,R,chi,L):
    par_curr = [parallel_current(omega,phi,j_val,m_val,R,chi,L) for j_val in range(L) for m_val in range(1,3)]
    norm = np.max(np.abs(par_curr))
    return parallel_current(omega,phi,j,m,R,chi,L)/norm

def parallel_current_homogeneous(omega,phi,m,R,chi,L):
    if R< R_threshold(chi):
        a1 = -1/(L)*np.sin((-1)**(m-1)*chi/2 + k_plus(R,chi,L)) *np.cos(omega)**2*(1 + (-1)**m* np.cos(2*theta(k_plus(R,chi,L),R,chi)))
        #a1 = 0
        a2 = -1/(L)*np.sin((-1)**(m-1)*chi/2 - k_plus(R,chi,L) ) *np.sin(omega)**2*(1 + (-1)**(m-1)* np.cos(2*theta(k_plus(R,chi,L),R,chi))) 
        #a2 = 0
        return (a1 + a2)
    else:
        return (-1)**(m+1)*np.sin(chi/2)
def parallel_current_non_homogeneous(omega,phi,j,m,R,chi,L):
    if R<R_threshold(chi):
        return (-1)**(m+1)*1/(L)*np.sin(chi/2)*np.sin(2*omega)*np.sin(2*theta(k_plus(R,chi,L),R,chi))*np.cos(2*j*k_plus(R,chi,L) + k_plus(R,chi,L) -phi)
        
    else:
        return 0
        
def vortex_currents_multi_chi(omega, phi, R, chi_values, L):
    fig, ax = plt.subplots(figsize=(10, 8))
    norm = plt.Normalize(vmin=-1, vmax=1) 
    cmap = cm.coolwarm  
    #cmap = plt.get_cmap('viridis')
    for idx, chi in enumerate(chi_values):
        y_offset = idx * 3  
        x = np.arange(1, L + 1)
        y1 = np.full(L, y_offset + 2)  
        y2 = np.full(L, y_offset + 1)  
        currents_horiz_bas = [parallel_current_normalized(omega, phi, j, 2, R, chi, L) for j in range(L)]
        currents_horiz_haut = [parallel_current_normalized(omega, phi, j, 1, R, chi, L) for j in range(L)]
        currents_vert = [perp_current_normalized(omega, phi, j, R, chi, L) for j in range(L)]
        ax.plot(x, y1, 'bo')  
        ax.plot(x, y2, 'ro')
        for i in range(L - 1):
            ax.plot([x[i], x[i+1]], [y1[i], y1[i]], color=cmap(norm(currents_horiz_bas[i])), linewidth=5)
            ax.plot([x[i], x[i+1]], [y2[i], y2[i]], color=cmap(norm(currents_horiz_haut[i])), linewidth=5)
        for i in range(L):
            ax.plot([x[i], x[i]], [y1[i], y2[i]], color=cmap(norm(currents_vert[i])), linewidth=5)
        chi_str = f'{round(chi/np.pi, 2):>4}π'
        label = f'R = {R:.1f}, χ = {chi_str}'
        ax.text(-2, y_offset + 1.5, label, ha='right', va='center', fontsize=20, family='monospace')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Normalized non homogeneous current amplitude', fraction=0.035, pad=0.04)
    cbar.ax.yaxis.label.set_size(15)
    ax.set_aspect('equal')
    ax.set_xlabel('j', fontsize=15)
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    plt.title('Current patterns from Meissner to Vortex', fontsize=30)
    plt.tight_layout()
    plt.show()

#print('kmin', k_plus(0.5, np.pi * 0.35, 24) * 24 / (2 * np.pi))  # debug print

#vortex_currents_multi_chi( omega=np.pi/4, phi=np.pi/2,  R=0.5, chi_values=[np.pi*0.99, np.pi*0.4, np.pi*0.35, np.pi*0.2],   L=4*6)
def n_MF_plus(m,R,chi,L):
    return np.cos(theta(k_plus(R,chi,L),R,chi) + np.pi/2*(2-m))**2/L
def n_MF_minus(m,R,chi,L):
    return np.cos(theta(-k_plus(R,chi,L),R,chi) + np.pi/2*(2-m))**2/L

def n_MF_mix(j,m,omega,phi,R,chi,L):
    if R >= R_threshold(chi):
        return 1/(2*L)
    cross_term = -np.sin(2*omega)*np.sin(2*theta(k_plus(R,chi,L),R,chi))*np.cos(2*j*k_plus(R,chi,L) - phi)/(2*L)
    #return np.cos(omega)**2*n_MF_plus(m,R,chi,L) + np.sin(omega)**2*n_MF_minus(m,R,chi,L)  + cross_term
    return cross_term

def n_MF_mix_normalized(j,m,omega,phi,R,chi,L):
    norm = np.max(np.abs([n_MF_mix(j_val,m_val,omega,phi,R,chi,L) for j_val in range(L) for m_val in range(1,3)]))
    return n_MF_mix(j,m,omega,phi,R,chi,L)/norm
def plot_density_quarters(phi, omega, R, chi_values, L):
    fig, ax = plt.subplots(figsize=(10, 8))

    cmap = plt.get_cmap('viridis')
    norm = plt.Normalize(vmin=-1, vmax=1)
    all_densities = [( [n_MF_mix_normalized(j, 1, omega, phi, R, chi, L) for j in range(1, L + 1)],  [n_MF_mix_normalized(j, 2, omega, phi, R, chi, L) for j in range(1, L + 1)] )for chi in chi_values  ]
    for idx, (chi, (densities_leg1, densities_leg2)) in enumerate(zip(chi_values, all_densities)):
        y_offset = idx * 3
        x = np.arange(1, L + 1)
        y1 = np.full(L, y_offset + 2)
        y2 = np.full(L, y_offset + 1)
        for i in range(L - 1):
            x_center = (x[i] + x[i + 1]) / 2
            y_center = (y1[i] + y2[i]) / 2
            quadrants = [
                (densities_leg1[i],      [x[i], x_center, x_center, x[i]], [y1[i], y1[i], y_center, y_center]),
                (densities_leg1[i + 1],  [x_center, x[i + 1], x[i + 1], x_center], [y1[i], y1[i], y_center, y_center]),
                (densities_leg2[i],      [x[i], x_center, x_center, x[i]], [y_center, y_center, y2[i], y2[i]]),
                (densities_leg2[i + 1],  [x_center, x[i + 1], x[i + 1], x_center], [y_center, y_center, y2[i], y2[i]]) ]
            for density, xq, yq in quadrants:
                ax.fill(xq, yq, color=cmap(norm(density)), linewidth=0, antialiased=True)
        ax.plot(x, y1, 'ko', markersize=6)
        ax.plot(x, y2, 'ko', markersize=6)
        for i in range(L - 1):
            ax.plot([x[i], x[i + 1]], [y1[i], y1[i]], 'k-', linewidth=1.5)
            ax.plot([x[i], x[i + 1]], [y2[i], y2[i]], 'k-', linewidth=1.5)
        for i in range(L):
            ax.plot([x[i], x[i]], [y1[i], y2[i]], 'k-', linewidth=1.5)
        chi_str = f'{round(chi/np.pi, 2):>4}π'
        ax.text(-2, y_offset + 1.5, f'R = {R:.1f}, χ = {chi_str}',
                ha='right', va='center', fontsize=20, family='monospace')
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Normalized non homogeneous density amplitude', fraction=0.035, pad=0.04)
    cbar.ax.yaxis.label.set_size(15)
    ax.set_aspect('equal')
    ax.set_xlabel('j', fontsize=15)
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    plt.title('Density patterns from Meissner to Vortex', fontsize=30)
    ax.set_xlim(0.5, L + 0.5)
    ax.set_ylim(0, len(chi_values) * 3 + 2)
    plt.tight_layout()
    plt.show()

def R_lim(chi):
    return 2*np.sqrt(2)*np.sin(chi/2)**2/np.sqrt(2*np.cos(chi/2)**2 + 1)

#plot_density_quarters(phi=0,omega=0,R=0.5, chi_values=[np.pi*0.99, np.pi*0.4, np.pi*0.35, np.pi*0.2],L=24)
#plot_density_quarters(phi=0,omega=np.pi/4,R=0.5, chi_values=[np.pi*0.99, np.pi*0.4, np.pi*0.35, np.pi*0.2],L=24)
def plot_chiral_current_and_k_vs_R(k_vals, R_vals, chi_fixed, U, n, J_parallel, L):
    """
    Plot chiral current and minimizing k as functions of R for fixed chi and U,
    using points instead of continuous lines, with phase-colored backgrounds.
    """
    chiral_current_list = []
    k_min_list = []
    phase_list = []
    for R in R_vals:
        k_min, phase = get_minimizing_k_and_phase(k_vals, chi_fixed, R, U, n)
        current = chiral_crrent(k_min, R, chi_fixed, L)
        chiral_current_list.append(current)
        k_min_list.append(k_min)
        phase_list.append(phase)
    chiral_current_list = np.array(chiral_current_list)
    k_min_list = np.array(k_min_list)
    phase_colors = {'Meissner': 'red',  'Biased Ladder': 'purple', 'Vortex': 'green' }
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    prev_phase = phase_list[0]
    start_idx = 0
    for idx, phase in enumerate(phase_list):
        if phase != prev_phase or idx == len(phase_list) - 1:
            if idx == len(phase_list) - 1:
                idx += 1 
            for ax in axs:
                ax.axvspan(R_vals[start_idx], R_vals[idx-1], color=phase_colors.get(prev_phase, 'gray'), alpha=0.3 )
            prev_phase = phase
            start_idx = idx
    norm = np.max(np.abs(chiral_current_list))
    axs[0].scatter(R_vals, chiral_current_list/norm, s=30, color='blue') 
    axs[0].set_ylabel(r'$\frac{\langle\hat{J_c}\rangle}{J^0_c}$', fontsize=28)
    axs[0].set_title(f'$\chi={chi_fixed:.2f}$, $U={U}$, $n={n}$', fontsize=26)
    axs[0].grid(True)
    axs[0].tick_params(labelsize=20)
    axs[1].scatter(R_vals, k_min_list, s=30, color='blue')  
    axs[1].set_xlabel(r'$R$', fontsize=30)
    axs[1].set_ylabel(r'$k_{\mathrm{min}}$', fontsize=28)
    axs[1].grid(True)
    axs[1].tick_params(labelsize=20)
    legend_elements = [
        Patch(facecolor='red', edgecolor='r', label='Meissner'),
        Patch(facecolor='purple', edgecolor='purple', label='Biased ladder'),
        Patch(facecolor='green', edgecolor='green', label='Vortex') ]
    axs[0].legend(handles=legend_elements, loc='best', fontsize=18)
    plt.tight_layout()
    plt.show()
#plot_chiral_current_and_k_vs_R(k_vals=np.linspace(0, np.pi, 3000),  R_vals=np.linspace(0.001, 2, 1000),  chi_fixed=np.pi/2, U=0.005,  n=0.5,  J_parallel=1,L=3000)
def create_chiral_current_vs_R_and_U(k_vals, R_vals, U_vals, chi_fixed, n, J_parallel, L):
    start_t = time.time()
    """Create a color plot of the normalized chiral current across R and U, 
    with fixed chi using mean-field minimizing k and omega."""
    chiral_current_grid = np.zeros((len(R_vals), len(U_vals)))
    for i, R in enumerate(R_vals):
        for j, U_val in enumerate(U_vals):
            print('R, U', R, U_val)
            print('time', time.time() - start_t)
            k_min, phase = get_minimizing_k_and_phase(k_vals, chi_fixed, R, U_val, n)
            current = chiral_crrent(k_min, R, chi_fixed, L)
            chiral_current_grid[i, j] = current
    norm = np.max(np.abs(chiral_current_grid))
    plt.figure(figsize=(10, 6))
    plt.imshow(
        chiral_current_grid / norm, extent=[min(U_vals), max(U_vals), min(R_vals), max(R_vals)],   origin='lower',   aspect='auto',  cmap='viridis' ) 
    cbar = plt.colorbar()
    cbar.set_label(r'$\frac{\langle\hat{J_c}\rangle}{J^0_c}$', fontsize=24)
    cbar.ax.tick_params(labelsize=20)
    plt.xlabel(r'$U$', fontsize=30)
    plt.ylabel(r'$R$', fontsize=30)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title(r'$\chi = \frac{\pi}{2}$'+f', $n = {n}$', fontsize=30)
    plt.tight_layout()
    plt.show()
    return chiral_current_grid

#chiral_current_grid = create_chiral_current_vs_R_and_U(   k_vals=np.linspace(0, np.pi, 200),   R_vals=np.linspace(0.7, 1.5, 400),   U_vals=np.linspace(0.001, 0.8, 200),    chi_fixed=np.pi/2,    n=0.5,    J_parallel=1,   L=200)


def create_density_imbalance_phase_diagram(k_vals, R_vals, chi_vals, U, n, J_parallel, L):
    start_t = time.time()
    """Create a color plot of the density imbalance across R and chi."""
    density_imbalance_grid = np.zeros((len(R_vals), len(chi_vals)))  
    for i, R in enumerate(R_vals):   
        for j, chi in enumerate(chi_vals): 
            print('R, chi', R, chi)
            print('time', time.time() - start_t)
            k_min, phase = get_minimizing_k_and_phase(k_vals, chi, R, U, n)
            if phase == "Vortex":  
                omega = np.pi / 4
            else:  
                omega = 0
            phi = 0  
            psi = psi_mix_k_new(k_min, R, chi, L, omega, phi)  
            density_imbalance = density_imbalance_y(psi, L)
            density_imbalance_grid[i, j] = density_imbalance
    norm = np.max(np.abs(density_imbalance_grid))
    plt.figure(figsize=(10, 6))
    plt.imshow(density_imbalance_grid / norm, extent=[min(chi_vals), max(chi_vals), min(R_vals), max(R_vals)], origin='lower', aspect='auto', cmap='viridis')
    R_line = R_lim(np.linspace(0.01, 2.25, 1000))
    plt.plot(np.linspace(0, 2.25, 1000), R_line, color='red', lw=5, label=r'$R_{lim}(\chi)$')
    cbar = plt.colorbar()
    cbar.set_label(r'$\frac{\Delta n}{n^0}$', fontsize=24) 
    cbar.ax.tick_params(labelsize=20) 
    plt.xlabel(r'$\chi$', fontsize=30)
    plt.ylabel(r'$R$', fontsize=30)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title(f'U = {U}, n = {n}', fontsize=30)
    plt.tight_layout()
    plt.legend(fontsize=20)
    plt.show()
    return density_imbalance_grid

#density_imbalance_grid = create_density_imbalance_phase_diagram( np.linspace(0, np.pi, 200), np.linspace(0.001, 2, 400),   np.linspace(0, np.pi*0.99, 300),   U=0.005, n=0.5, J_parallel=1, L=200)


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

#L=20
#omega = np.pi/4
#J_perp = 0.05
#R = J_perp/J_parallel
#U = 0.01
#fluc_max = 0

def plot_psi(psi,R,chi,U, L):
    psi = psi.reshape(L, 2)
    magnitudes = np.abs(psi)  
    phases = np.angle(psi) 
    fig, ax = plt.subplots(figsize=(12, 6))
    norm_mag = plt.Normalize(vmin=np.min(magnitudes)*0.999, vmax=np.max(magnitudes)*1.001)
    norm_phase = plt.Normalize(vmin=-np.pi, vmax=np.pi)
    y_mag = 0  
    for m in range(1,3):  
        y_offset = y_mag + (2-m) * 0.4  
        for j in range(L): 
            color = plt.cm.viridis(norm_mag(magnitudes[j, m-1]))
            ax.fill(
                [j+1,j+2,j+2,j+1],
                [y_offset, y_offset, y_offset+0.4, y_offset+0.4],
                color=color
            )
    y_phase = -1  
    for m in range(1,3):  
        y_offset = y_phase + (2-m)*0.4 
        for j in range(L): 
            color = plt.cm.twilight(norm_phase(phases[j, m-1]))
            ax.fill(
                [j+1, j+2,j+2,j+1],
                [y_offset, y_offset, y_offset + 0.4, y_offset + 0.4],
                color=color
            )
    sm_mag = plt.cm.ScalarMappable(cmap='viridis', norm=norm_mag)
    sm_phase = plt.cm.ScalarMappable(cmap='twilight', norm=norm_phase)
    sm_mag.set_array([])
    sm_phase.set_array([])
    cbar_mag = plt.colorbar(sm_mag, ax=ax, label="Magnitude",pad=0.1)
    cbar_phase = plt.colorbar(sm_phase, ax=ax, label="Phase", pad=0.1)
    ax.axhline(0, color='black', lw=1) 
    ax.set_xlim(0.5,L+0.5)
    ax.set_ylim(-1.2, 0.8)
    ax.set_xticks(range(1,L+1))
    ax.set_yticks([y_mag+0.2, y_mag+0.6, y_phase+0.2, y_phase+0.6])
    ax.set_yticklabels(["m=2", "m=1", "m=2", "m=1"])
    ax.set_title(f"R={R}, chi= {chi},U={U},L={L}")
    plt.xlabel("j")
    plt.tight_layout()
    plt.show()

def convergence_test(L,R,chi,U,N,J_parallel =1, step = 0.1, n_steps = 1000, t0=0,phi =0,fluc_max = 0):
    J_perp = R
    time_start = time.time()
    energies = []
    final_states = []
    for i in range(-L//2,L//2):
        omega1 = np.pi/4
        omega2 = np.pi/2
        omega1, cond1 = initial_cond_k(2*np.pi*i/L,omega1,fluc_max,R,chi, L,phi)
        omega2, cond2 = initial_cond_k(2*np.pi*i/L,omega2,fluc_max,R,chi, L,phi)
        omega,y0,final_state = compare_energy(cond1,omega1,cond2,omega2,R,chi,U,L,N,t0,step,n_steps)
        final_states = final_states + [[i,omega,final_state]]
        E =  Energy_functional(final_state,J_parallel,J_perp,chi,U,L)
        energies = energies + [E]
    best_state_index = np.argmin(energies)
    k,omega, final_state = final_states[best_state_index]
    print('the right k and omega =',k,omega)
    #omega = np.pi/4
    y0 = np.zeros(2*L, dtype=complex)  
    for j in range(L):
        y0[2*j] = psi_mix_k(2*np.pi*k/L,j,1,R,chi,L,omega,phi)  
        y0[2*j + 1] = psi_mix_k(2*np.pi*k/L,j,2,R,chi,L,omega,phi)
        #y0[2*j] = sum(psi_mix_k(2*np.pi*k/L + 0.1,j,1,R,chi,L,omega,phi)  for  k in range(L))
        #y0[2*j + 1] = sum(psi_mix_k(2*np.pi*k/L + 0.1,j,2,R,chi,L,omega,phi)  for k in range(L))
    
    t_values, y_values = RK4(differential_equations, t0, y0, step, n_steps,J_parallel,J_perp,chi,U,N)
    y_values_reshaped = y_values.reshape((L, 2, n_steps))
    #print('y_vals abs',np.abs(y_values_reshaped))
    energy =[Energy_functional(y,J_parallel,J_perp,chi,U,L) for y in y_values]
    density_imbalance_alongx = [vortex_density(y,L) for y in y_values]
    density_imbalance_alongy = [density_imbalance_y(y,L) for y in y_values]
    chiral_current = [chiral_current_functional(y,J_parallel,chi,L) for y in y_values]
    print(' final state energy ',energy[-1])
    print('k,maxk,density imbalance,  vortex density, chiral current, ',k, int(L/(2*np.pi)*k_plus(R,chi,L)), density_imbalance_alongy[-1], density_imbalance_alongx[-1],chiral_current[-1])
    time_end= time.time()
    print('temps',time_end - time_start)

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))  

    axs[0, 0].plot(t_values[10:], energy[10:], lw=4)
    axs[0, 0].set_ylabel(r'Energy (units of $J_{\parallel}$)', fontsize=20)
    axs[0, 0].set_xlabel('time steps (logscale)', fontsize=20)
    axs[0, 0].set_xscale('log')
    axs[0, 0].tick_params(axis='both', labelsize=15)

    axs[0, 1].plot(t_values[10:], density_imbalance_alongy[10:], lw=4)
    axs[0, 1].set_ylabel(r'Imbalance along y (units of $J_{\parallel}$)', fontsize=20)
    axs[0, 1].set_xlabel('time steps (logscale)', fontsize=20)
    axs[0, 1].set_xscale('log')
    axs[0, 1].tick_params(axis='both', labelsize=15)

    axs[1, 0].plot(t_values[10:], density_imbalance_alongx[10:], lw=4)
    axs[1, 0].set_ylabel(r'Vortex density', fontsize=20)
    axs[1, 0].set_xlabel('time steps (logscale)', fontsize=20)
    axs[1, 0].set_xscale('log')
    axs[1, 0].tick_params(axis='both', labelsize=15)

    axs[1, 1].plot(t_values[10:], chiral_current[10:], lw=4)
    axs[1, 1].set_ylabel(r'Chiral current (units of $J_{\parallel}$)', fontsize=20)
    axs[1, 1].set_xlabel('time steps (logscale)', fontsize=20)
    axs[1, 1].set_xscale('log')
    axs[1, 1].tick_params(axis='both', labelsize=15)

    plt.tight_layout()
    plt.show()
    psi = y_values[-1].reshape((L, 2))
    plot_psi(psi,R,chi,U ,L)

#convergence_test(20,1.2,np.pi/2,0.2,20,1,0.01,1000)

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

psi_winding = psi_mix_k_new(2*np.pi/100,0.5,np.pi/2,100,np.pi/4,0)
b,c,psi_winding = GPE_condensate(0.5,np.pi/2,0.1,1/2,100,100,1,0,0,0,0.01,10)
#print('vortex density1', vortex_density(psi_winding,100))
#print('vortex density2', vortex_density_new(psi_winding,100))
#L= 10
#R = 0.5
#b,c,d =GPE_condensate(R,np.pi/2,0.1,1/2,L,L,1,0,0,0,0.01,1)
#nb_bosons = (np.conj(d)@d).real
#_,target_psi = initial_cond_k(b*2*np.pi/L,c,0,R,np.pi/2, L)
#target_psi = target_psi
#print('GPE condensate and N and b and c', np.abs(d), nb_bosons,b,c)
#print('target psi and particle number', np.abs(target_psi*np.sqrt(L)), np.conj(target_psi)@target_psi)
#mu_GPE = (Kinetic(d,1,0.5,np.pi/2) + 2* Interac(d,1,R,0.1))/nb_bosons
#print('mu GPE',mu_GPE)

#psik = psi_k(b*2*np.pi/L,R,np.pi/2,L)
#psik_prime = psi_k(b*2*np.pi/L,R,np.pi/2,L)
#om,phi = np.pi/4, 0
#psi_mix = psi_mix_k_new(b*2*np.pi/L,R,np.pi/2,L,om,phi)
#psi_mix_compare = np.zeros(2*L, dtype = complex)
#for site in range(L):
#    psi_mix_compare[2*site ] = psi_mix_k(b*2*np.pi/L,site,1,R,np.pi/2,L,om,phi)
#    psi_mix_compare[2*site +1 ] = psi_mix_k(b*2*np.pi/L,site,2,R,np.pi/2,L,om,phi)
#_,psi_mix_compare2 =   initial_cond_k(b*2*np.pi/L,om,0,R,np.pi/2, L)
#vort1 =vortex_density(psi_mix,L)
#vort2 =vortex_density(psi_mix_compare,L)
#vort3 =vortex_density(psi_mix_compare2,L)
#print('vort1,2,3', vort1,vort2,vort3)
#print('diff psi', np.max(np.abs(psi_mix_compare - psi_mix)))
#print('diff psi 2', np.max(np.abs(psi_mix_compare2 - psi_mix)))
#print('psi mix norm', np.conj(psi_mix)@psi_mix)
#print('psi k norm', np.conj(psik)@psik)
#print('psi k orthogonality', np.conj(psik)@psik_prime)



def phase_diagram_data_k_old(R_range, U_range, R_points, U_points, chi, L, t0, step, n_steps, fluc_max=0.0, phi=0):
    time_start = time.time()
    J_parallel = 1 
    R_cell_size = (R_range[1] - R_range[0]) / R_points
    U_cell_size = (U_range[1] - U_range[0]) / U_points                           
    R_values = np.linspace(*R_range, R_points) + R_cell_size / 2
    U_values = np.linspace(*U_range, U_points) + U_cell_size / 2
    file_name = f"optimized_L{L}_phase_data_chi_{round(chi,2)}.csv"
    with open(file_name, "w") as f:
        f.write("U,R,Density_Imbalance,V_density,Chiral_Current, Omega,k,final state\n")
        for R in R_values:
            for U in U_values:
                J_perp = R
                energies = []
                final_states = []
                for k in range(0,L//2):
                    omega1 = np.pi/4
                    omega2 = np.pi/2
                    omega1, cond1 = initial_cond_k(2*np.pi*k/L,omega1,fluc_max,R,chi, L,phi)
                    omega2, cond2 = initial_cond_k(2*np.pi*k/L,omega2,fluc_max,R,chi, L,phi)
                    omega,y0,final_state = compare_energy(cond1,omega1,cond2,omega2,R,chi,U,L,N,t0,step,n_steps)
                    final_states = final_states + [[k,omega,final_state]]
                    E =  Energy_functional(final_state,J_parallel,J_perp,chi,U,L)
                    energies = energies + [E]
                best_state_index = np.argmin(energies)
                k,omega, final_state = final_states[best_state_index]
                imbalance_y = density_imbalance_y(final_state, L)
                Vortex_nb = vortex_density(final_state, L)
                chiral_current = chiral_current_functional(final_state, J_parallel, L)
                f.write(f"{U},{R},{imbalance_y},{Vortex_nb},{chiral_current},{omega},{k},{final_state}\n")
                time_end = time.time()
                print('omega = ', omega, ' k = ', k)
                print(time_start - time_end)
    total_time = time_start - time.time()
    with open(file_name, "a") as f:
        f.write(f"# Total simulation time: {-total_time:.2f} seconds \n")
        f.write(f"# Parameters: chi={chi}, L={L}, R_range : from {R_range[0]} to {R_range[-1]} , U_range : from {U_range[0]} to {U_range[-1]}, R_points={R_points}, U_points={U_points}, step={step}, n_steps={n_steps}, fluc_max={fluc_max}, phi={phi}\n")


def phase_diagram_data_k(R_range, U_range, R_points, U_points, chi, L, t0, step, n_steps, fluc_max=0.0, phi=0): #cluster friendly
    time_start = time.time()
    J_parallel = 1
    R_cell_size = (R_range[1] - R_range[0]) / R_points
    U_cell_size = (U_range[1] - U_range[0]) / U_points                           
    R_values = np.linspace(*R_range, R_points) + R_cell_size / 2
    U_values = np.linspace(*U_range, U_points) + U_cell_size / 2

    file_name = f"optimized_L{L}_phase_data_chi_{round(chi,2)}_nsteps_{n_steps}_resolution_{R_points*U_points}.csv"

    with open(file_name, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["U", "R", "Density_Imbalance", "Vortex_Density", "Chiral_Current", "Omega", "k", "Final_State"])
        start_time = time.time()
        for R in R_values:
            for U in U_values:
                J_perp = R
                energies = []
                final_states = []
                for k in range(0, L//2):
                    omega1 = np.pi/4
                    omega2 = np.pi/2
                    omega1, cond1 = initial_cond_k(2*np.pi*k/L, omega1, fluc_max, R, chi, L, phi)
                    omega2, cond2 = initial_cond_k(2*np.pi*k/L, omega2, fluc_max, R, chi, L, phi)
                    omega, y0, final_state = compare_energy(cond1, omega1, cond2, omega2, R, chi, U, L,N,t0,step,n_steps)
                    final_states.append([k, omega, final_state])
                    E = Energy_functional(final_state, J_parallel, J_perp, chi, U, L)
                    energies.append(E)
                best_state_index = np.argmin(energies)
                k, omega, final_state = final_states[best_state_index]
                imbalance_y = density_imbalance_y(final_state, L)
                vortex_nb = vortex_density(final_state, L)
                chiral_current = chiral_current_functional(final_state, J_parallel,chi, L)

                final_state_str = ' '.join(map(str, final_state.flatten()))
                writer.writerow([U, R, imbalance_y, vortex_nb, chiral_current, omega, k, final_state_str])
                elapsed_time = time.time() - start_time
                print(f"Processed U={U}, R={R}, omega={omega}, k={k} in {elapsed_time:.2f} seconds.")
    total_time = round(time.time() - time_start, 2)
    with open(file_name, "a", newline='') as f:
        f.write(f"# Total simulation time: {total_time} seconds\n")
        f.write(f"# Parameters: chi={chi}, L={L}, R_range: from {R_range[0]} to {R_range[1]}, U_range: from {U_range[0]} to {U_range[1]}, R_points={R_points}, U_points={U_points}, fluc_max={fluc_max}, phi={phi}\n")
   
#phase_diagram_data_k(R_range, U_range, R_points, U_points, chi, L, t0, step, n_steps, fluc_max=0.0, phi=0)

def phase_diagram_from_data_old(filename):  #old version
    #chi_value = float(filename.split('_')[-1].split('.')[0] + filename.split('_')[-1].split('.')[1])  /100   
    chi_value =  1.57                    
    data = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)  
        for row in reader:
            #print('cest row !', row)
            #print('j' in row[0],'le premier el', row[0])
            if row[0].startswith('#') or 'j' in row[0]:
                print('skipped')
                continue
            U = float(row[0])
            R = float(row[1])
            density_imbalance = float(row[2])
            vortex_nb = float(row[3])
            chiral_current = float(row[4])
            data.append([U, R, density_imbalance, vortex_nb, chiral_current])
    
    U_values = sorted(set(row[0] for row in data))
    R_values = sorted(set(row[1] for row in data))

    U_grid, R_grid = np.meshgrid(U_values, R_values)

    imbalance_grid = np.zeros_like(U_grid)
    vortex_grid = np.zeros_like(U_grid)
    chiral_current_grid = np.zeros_like(U_grid)

    for row in data:
        U = row[0]
        R = row[1]
        density_imbalance = row[2]
        vortex_nb = row[3]
        chiral_current = row[4]

        U_index = U_values.index(U)
        R_index = R_values.index(R)

        imbalance_grid[R_index, U_index] = density_imbalance
        vortex_grid[R_index, U_index] = vortex_nb
        chiral_current_grid[R_index, U_index] = chiral_current

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    im1 = ax[0].imshow(imbalance_grid, origin='lower', aspect='auto', cmap='viridis', extent=[U_values[0], U_values[-1], R_values[0], R_values[-1]])
    ax[0].set_title(r"$\chi$ =" + f"{round(chi_value,2)} " +'  Density Imbalance')
    ax[0].set_xlabel('U')
    ax[0].set_ylabel('R')
    fig.colorbar(im1, ax=ax[0])
 
    im2 = ax[1].imshow(vortex_grid, origin='lower', aspect='auto', cmap='plasma', extent=[U_values[0], U_values[-1], R_values[0], R_values[-1]])
    ax[1].set_title('Vortex Density')
    ax[1].set_xlabel('U')
    ax[1].set_ylabel('R')
    fig.colorbar(im2, ax=ax[1])

    im3 = ax[2].imshow(chiral_current_grid, origin='lower', aspect='auto', cmap='inferno', extent=[U_values[0], U_values[-1], R_values[0], R_values[-1]])
    ax[2].set_title('Chiral Current')
    ax[2].set_xlabel('U')
    ax[2].set_ylabel('R')
    fig.colorbar(im3, ax=ax[2])

    plt.tight_layout()
    plt.show()

def phase_diagram_from_data(filename): #cluster friendly
    chi = 1.57  
    data = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)  
        for row in reader:
            if row[0].startswith('#') or 'j' in row[0]:
                continue  
            U = float(row[0])
            R = float(row[1])
            density_imbalance = float(row[2])
            vortex_nb = float(row[3])
            chiral_current = float(row[4])
            data.append([U, R, density_imbalance, vortex_nb, chiral_current])
    
    U_values = sorted(set(row[0] for row in data))
    R_values = sorted(set(row[1] for row in data))
    U_grid, R_grid = np.meshgrid(U_values, R_values)
    
    imbalance_grid = np.zeros_like(U_grid)
    vortex_grid = np.zeros_like(U_grid)
    chiral_current_grid = np.zeros_like(U_grid)
    
    for row in data:
        U = row[0]
        R = row[1]
        density_imbalance = row[2]
        vortex_nb = row[3]
        chiral_current = row[4]
        
        U_index = U_values.index(U)
        R_index = R_values.index(R)
        
        imbalance_grid[R_index, U_index] = density_imbalance
        vortex_grid[R_index, U_index] = vortex_nb
        chiral_current_grid[R_index, U_index] = chiral_current

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    norm1 = np.max(np.abs(imbalance_grid))

    im1 = ax[0].imshow(imbalance_grid/norm1, origin='lower', aspect='auto', cmap='viridis', extent=[U_values[0], U_values[-1], R_values[0], R_values[-1]])
    #ax[0].set_title(r"$\chi$ = $\frac{\pi}{2}$", fontsize=18)  
    ax[0].set_xlabel('U', fontsize=25)
    ax[0].set_ylabel('R', fontsize=25)
    fig.colorbar(im1, ax=ax[0]).set_label(r'$\frac{\Delta n}{ n^0}$', fontsize=20)  
    
    im2 = ax[1].imshow(vortex_grid, origin='lower', aspect='auto', cmap='viridis', extent=[U_values[0], U_values[-1], R_values[0], R_values[-1]])
    #ax[1].set_title('Vortex Density', fontsize=16)
    ax[1].set_xlabel('U', fontsize=25)
    ax[1].set_ylabel('R', fontsize=25)
    fig.colorbar(im2, ax=ax[1]).set_label(r'$n_{vortex}$', fontsize=20)  
    norm2 = np.max(np.abs(chiral_current_grid))

    im3 = ax[2].imshow(chiral_current_grid/norm2, origin='lower', aspect='auto', cmap='viridis', extent=[U_values[0], U_values[-1], R_values[0], R_values[-1]])
    ax[2].set_title(r"$\chi = \frac{\pi}{2}, N = 100, L=200$", fontsize=30)
    ax[2].set_xlabel('U', fontsize=45)
    ax[2].set_ylabel('R', fontsize=45)
    ax[2].tick_params(axis='both', which='major', labelsize=35)  
    cbar = fig.colorbar(im3, ax=ax[2])
    cbar.set_label(r'$\frac{\langle \hat{J}_c \rangle}{J_c^0}$', fontsize=45)

    cbar.ax.tick_params(labelsize=35) 

    
    plt.suptitle(r"$\chi = \frac{\pi}{2}, N = 100$", fontsize=30)  
    plt.tight_layout()
    plt.subplots_adjust(top=0.85) 
    plt.show()

filename = "optimized_L20_phase_data_chi_1.57.csv" 
filename = "optimized_L80_phase_data_chi_1.57_nsteps_1000_resolution_2625.csv"
filename = "optimized_L200_phase_data_chi_1.57_nsteps_2000_resolution_6000.csv"
#filename = "optimized_L100_phase_data_chi_1.57_nsteps_6000_resolution_2625.csv"
#filename = "v2_optimized_L20_phase_data_chi_1.57_nsteps_1000_resolution_1225.csv"
#phase_diagram_from_data(filename)


def plot_density_profile(csv_file, fixed_U, fixed_R):
    df = pd.read_csv(csv_file, comment='#')
    df['distance'] = np.sqrt((df["U"] - fixed_U)**2 + (df["R"] - fixed_R)**2)
    closest_row = df.loc[df['distance'].idxmin()]
    state_str = closest_row["Final_State"]  
    state = np.array([complex(s) for s in state_str.strip("[]").split()])  
    density = np.abs(state)**2  
    L = len(density) // 2 
    upper_leg = density[::2] 
    lower_leg = density[1::2]  
    plt.plot(range(L), upper_leg, 'bo-', label="Upper leg", markersize=15)  
    plt.plot(range(L), lower_leg, 'r*-', label="Lower leg", markersize=15)  
    plt.xlabel("Site j",size = 30)
    plt.ylabel(r"$|\psi_{j,m}|^2$",size = 30)
    plt.title(f"(U, R) = ({closest_row['U']}, {round(closest_row['R'],2)})",size = 30)
    plt.legend(fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True)
    plt.show()
    return closest_row["U"], closest_row["R"]


filename = "optimized_L80_phase_data_chi_1.57_nsteps_1000_resolution_2625.csv"
fixed_U = 0.15
fixed_R = 1
#plot_density_profile(filename, fixed_U, fixed_R)

filename = "optimized_L200_phase_data_chi_1.57_nsteps_2000_resolution_150.csv"
fixed_U = 0.175
fixed_R = 1.0
#plot_density_profile(filename, fixed_U, fixed_R)


def density_imbalance_vs_R(csv_file, fixed_U,L): 
    df = pd.read_csv(csv_file, comment='#')
    df_filtered = df[df["U"] == fixed_U].sort_values(by="R")
    fig = plt.figure()
    ax2 = fig.add_subplot(111)
    ax2.plot(df_filtered["R"], df_filtered["Density_Imbalance"], marker='o', linestyle='None', color='red',markersize = 20)
    ax2.set_title(f"Density imbalance vs. R for U = {fixed_U}, "+r"$\chi = \frac{\pi}{2}$ "  + f" L = {L}",size = 30)
    ax2.set_xlabel("R",size = 30)
    ax2.set_ylabel(r"$\frac{|n_1 - n_2|}{n_1 + n_2}$ ",size = 30)
    ax2.grid()
    plt.show()


fixed_U = 0.1689075630252101
fixed_U = 0.175 
filename = "optimized_L200_phase_data_chi_1.57_nsteps_2000_resolution_150.csv"
#density_imbalance_vs_R(filename, fixed_U,200)

def plot_chiral_current_vs_R(csv_file, fixed_U,L): 
    df = pd.read_csv(csv_file, comment='#')
    print('first',print(df.head(50)) )  
    df_filtered = df[np.isclose(df["U"], fixed_U, atol=1e-4)].sort_values(by="R")

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    for i in range(len(df_filtered["R"])):
        print(df_filtered["R"].iloc[i], df_filtered["Chiral_Current"].iloc[i])
        #print('ok')
    norm = np.max(np.abs( df_filtered["Chiral_Current"]))
    ax1.plot(df_filtered["R"], df_filtered["Chiral_Current"]/norm, marker='o', linestyle='None', color='purple',markersize = 15)
    ax1.set_title(f"U = {fixed_U}, "+r"$\chi = \frac{\pi}{2}$ "  + f" L = {L},N= 100",size = 30)
    ax1.set_xlabel("R",size = 30)
    ax1.set_ylabel(r"$\frac{\langle\hat{J}_{c}\rangle}{J^0_c}$",size = 30)
    ax1.tick_params(axis='both', which='major', labelsize=20)

    ax1.grid()
    plt.show()


fixed_U = 0.1689075630252101
fixed_U = 0.175
filename = "optimized_L80_phase_data_chi_1.57_nsteps_1000_resolution_50.csv" 
#plot_chiral_current_vs_R(filename, fixed_U,80)
filename = "optimized_L20_phase_data_chi_1.57_nsteps_1000_resolution_150.csv"
fixed_U = 0.05
#plot_chiral_current_vs_R(filename, fixed_U,20)
filename = "optimized_L40_phase_data_chi_1.57_nsteps_1000_resolution_150.csv"
#plot_chiral_current_vs_R(filename, fixed_U,40)
filename = "optimized_L60_phase_data_chi_1.57_nsteps_1000_resolution_150.csv"
#plot_chiral_current_vs_R(filename, fixed_U,60)
filename = "optimized_L80_phase_data_chi_1.57_nsteps_1000_resolution_150.csv"
#plot_chiral_current_vs_R(filename, fixed_U,80)
filename = "optimized_L80_phase_data_chi_1.57_nsteps_1000_resolution_2625.csv"
#plot_chiral_current_vs_R(filename, fixed_U,80)
filename = "optimized_L200_phase_data_chi_1.57_nsteps_2000_resolution_150.csv"
#plot_chiral_current_vs_R(filename, fixed_U,200)
#filename = "optimized_L200_phase_data_chi_1.57_nsteps_2000_resolution_6000.csv"
fixed_U = 0.258404
fixed_U = 0.173658
fixed_U = 0.175
#plot_chiral_current_vs_R(filename, fixed_U,200)

#Bogolibov spectrum



def chiral_current_analytical(omega,j,m,phi,J_parallel,chi,R,L):
    part1 = J_parallel/L* np.sin(chi/2 + k_plus(R,chi,L))*(1 - np.cos(2*theta(k_plus(R,chi,L),R,chi)))
    part2 = J_parallel/L* np.sin(chi/2 - k_plus(R,chi,L))*(1 + np.cos(2*theta(k_plus(R,chi,L),R,chi)))
    part3 = -J_parallel/L*np.exp(2*1j*k_plus(R,chi,L)*j +1j*k_plus(R,chi,L))*np.sin(chi/2)*np.sin(2*theta(k_plus(R,chi,L),R,chi))
    result = np.cos(omega)**2*part1 + np.sin(omega)**2*part2 + np.cos(omega)*np.sin(omega)*np.exp(1j*phi)*part3 + np.cos(omega)*np.sin(omega)*np.exp(-1j*phi)*np.conj(part3)
    return result

def average_chiral_current_analytical(omega,m,phi,J_parallel,chi,R,L):
    summation = 0
    for j in range(L):
        summation += chiral_current_analytical(omega,j,m,phi,J_parallel,chi,R,L)
    return summation/L
def chiral_current_analytical_leg2(omega,j,m,phi,J_parallel,chi,R,L):
    part1 = J_parallel/L* np.sin(-chi/2 + k_plus(R,chi,L))*(1 + np.cos(2*theta(k_plus(R,chi,L),R,chi)))
    part2 = J_parallel/L* np.sin(-chi/2 - k_plus(R,chi,L))*(1 - np.cos(2*theta(k_plus(R,chi,L),R,chi)))
    part3 = +J_parallel/L*np.exp(2*1j*k_plus(R,chi,L)*j +1j*k_plus(R,chi,L))*np.sin(chi/2)*np.sin(2*theta(k_plus(R,chi,L),R,chi))
    result = np.cos(omega)**2*part1 + np.sin(omega)**2*part2 + np.cos(omega)*np.sin(omega)*np.exp(1j*phi)*part3 + np.cos(omega)*np.sin(omega)*np.exp(-1j*phi)*np.conj(part3)
    return result

def average_chiral_current_analytical_leg2(omega,m,phi,J_parallel,chi,R,L):
    summation = 0
    for j in range(L):
        summation += chiral_current_analytical_leg2(omega,j,m,phi,J_parallel,chi,R,L)
    return summation/L

#################################################################################################################################################################
########################## trying to find if J_c depend on omega or not #########################################################################################
#################################################################################################################################################################
#phi=0
#chi = np.pi/2
#J_parallel = 1
#m= 1
#omega = np.pi/4
#omegas = np.linspace(0,np.pi/2,1000)

#print('average leg1 omega = pi/4',  average_chiral_current_analytical(omega,m,phi,J_parallel,chi,R,L))
#print('average leg2 omega = pi/4',  average_chiral_current_analytical_leg2(omega,m,phi,J_parallel,chi,R,L))
#print('sum chiral c omega = pi/4', average_chiral_current_analytical(omega,m,phi,J_parallel,chi,R,L) - average_chiral_current_analytical_leg2(omega,m,phi,J_parallel,chi,R,L))

#omega = 0
#print('average leg1 omega = 0',  average_chiral_current_analytical(omega,m,phi,J_parallel,chi,R,L))
#print('average leg2 omega = 0',  average_chiral_current_analytical_leg2(omega,m,phi,J_parallel,chi,R,L))
#print('sum chiral c omega = 0', average_chiral_current_analytical(omega,m,phi,J_parallel,chi,R,L) - average_chiral_current_analytical_leg2(omega,m,phi,J_parallel,chi,R,L))

#chiral_c = [chiral_current_analytical(x,chi,R,L) for x in omegas]
#print("analytical formula",chiral_current_ideal(omega,chi,R,L))
#psi1 = []
#psi2 = []
#for j in range(L):
#    for m in range(1,3):
#        psi1.append(psi_mix_k(k_plus(R,chi,L),j,m,R,chi,L,np.pi/4,phi))
#        psi2.append(psi_mix_k(k_plus(R,chi,L),j,m,R,chi,L,0,phi))
#norm1 = np.conj(psi1)@psi1
#norm2 = np.conj(psi2)@psi2

#psi1 = np.array(psi1/np.sqrt(norm1))
#psi2 = np.array(psi2/np.sqrt(norm2))

#chi = np.pi/2

#print("average leg1 functional omega = pi/4", average_parallel_current_functional_leg1(psi1,J_parallel,L))
#print("average leg2 functional omega = pi/4", average_parallel_current_functional_leg2(psi1,J_parallel,L))
#print("chiral current brute force  omega = pi/4 " ,chiral_current_functional(psi1,1,chi,L))


#print("average leg1 functional omega = 0", average_parallel_current_functional_leg1(psi2,J_parallel,L))
#print("average leg2 functional omega = 0", average_parallel_current_functional_leg2(psi2,J_parallel,L))
#print("chiral current brute force  omega = 0 " ,chiral_current_functional(psi2,1,chi,L))

#omegas = np.linspace(0,np.pi/2,5)
#chiral_c_functional = []
#chiral_c = []
#for x in omegas:
#    continue
#    psix = []
#    for j in range(L):
#        for m in range(1,3):
#            psix.append(psi_mix_k(k_plus(R,chi,L),j,m,R,chi,L,x,phi))
#    psix = np.array(psix)
#    sites = np.arange(L)
#    odd_psi_abs = [abs(psix[2*i]) for i in range(L)]
#    even_psi_abs = [abs(psix[2*i +1]) for i in range(L)]
#    plt.plot(sites, odd_psi_abs)
#    plt.plot(sites, even_psi_abs)
#    plt.title(f'omega = {x}')
#    plt.show()
#    chiral_c_functional.append(chiral_current_functional(psix,1,L))
#    chiral_c.append(average_chiral_current_analytical(x,m,phi,J_parallel,chi,R,L) - average_chiral_current_analytical_leg2(x,m,phi,J_parallel,chi,R,L))
#print('chiral c func',chiral_c_functional)
#print('chiral c func',chiral_c)
#plt.plot(omegas,chiral_c_functional,'r')
#plt.plot(omegas,chiral_c,'b')
#plt.show()


##################################################################################################################################################################
############################################################### chemical potential check  ########################################################################
##################################################################################################################################################################

#make sure that i recover mu 
def Energy_functional_with_mu(psis,J_parallel,J_perp,chi,U,L,mu):
    psis = psis.reshape(L,2)
    psis_right = np.roll(psis, shift=-1, axis=0)
    kin1 = -J_parallel*(np.exp(1j*chi*(1-p))*np.conjugate(psis[:, 0])@psis_right[:, 0]+np.exp(-1j*chi*(1-p))*psis[:, 0]@np.conjugate(psis_right[:, 0]))
    kin2 = -J_parallel * (np.exp(-1j*chi*p)*np.conjugate(psis[:, 1])@psis_right[:, 1] + np.exp(1j*chi*p)*psis[:, 1]@np.conjugate(psis_right[:, 1]))
    kin3 = -J_perp * (np.conjugate(psis[:, 0]) @ psis[:, 1] + np.conjugate(psis[:, 1]) @ psis[:, 0])
    interaction = (U/2)*(np.sum(abs(psis[:, 0])**4)   + np.sum(abs(psis[:, 1])**4) ) -mu*np.conjugate(psis[:, 0]) @ psis[:, 0] - mu*np.conjugate(psis[:, 1]) @ psis[:, 1]
    summation = kin1 + kin2 + kin3 + interaction
    return summation.real

def compare_energy_with_mu(cond1,omega1,cond2,omega2,R,chi,U,L,mu,t0,step,n_steps):
    J_parallel = 1
    J_perp = R
    t_values1, y_values1 = RK4_with_mu(differential_equations_with_mu, t0, cond1, step, n_steps, J_parallel, J_perp, chi,U,mu)
    t_values2, y_values2 = RK4_with_mu(differential_equations_with_mu, t0, cond2, step, n_steps, J_parallel, J_perp,chi, U,mu)
    final_state1 = y_values1[-1]
    final_state2 = y_values2[-1]
    Energy1 = Energy_functional_with_mu(final_state1,J_parallel,J_perp,chi,U,L,mu)
    Energy2 = Energy_functional_with_mu(final_state2,J_parallel,J_perp,chi,U,L,mu)
    if Energy1<Energy2:
        return omega1,cond1,final_state1
    return omega2,cond2,final_state2

def RK4_with_mu(f,t0,y0,step,n_steps,J_parallel,J_perp,chi,U,mu):
    y0 = normalized(y0,20)
    t_values = [t0]
    y_values = [y0]
    for i in range(1,n_steps):
        t_values.append(t_values[-1] + step)
        k1 = f(y_values[-1],t_values[-1],J_parallel,J_perp,chi,U,mu)
        k2 = f( y_values[-1] +step/2*k1, t_values[-1] + step/2 ,J_parallel,J_perp,chi,U,mu)
        k3 = f( y_values[-1] +step/2*k2, t_values[-1] + step/2 ,J_parallel,J_perp,chi,U,mu)
        k4 = f( y_values[-1] +step*k3, t_values[-1] + step ,J_parallel,J_perp,chi,U,mu)
        new_y_value = y_values[-1] + step/6*(k1 +2*k2 +2*k3 +k4)
        y_values.append(np.array(new_y_value))
    return np.array(t_values), np.array(y_values)

def differential_equations_with_mu(y,t,J_parallel,J_perp,chi,U,mu) : 
    """ y is a row vector of the form psi_{1,1}, psi_{1,2}, psi_{2,1}, psi_{2,2}, psi_{3,1}, psi_{3,2},... """
    y = np.array(y)
    L = len(y)//2
    psi = y.reshape((L, 2))
    dydt = np.zeros_like(psi, dtype=complex)
    psi_left = np.roll(psi, shift=1, axis=0)
    psi_right = np.roll(psi, shift=-1, axis=0)
    dydt[:, 0] = +J_parallel*(np.exp(-1j*chi*(1-p))*psi_left[:, 0] + np.exp(1j*chi*(1-p))*psi_right[:, 0]) + J_perp*psi[:, 1] - (U/2)*(2*abs(psi[:, 0])**2 ) * psi[:, 0] + mu*psi[:, 0]
    dydt[:, 1] = +J_parallel*(np.exp(1j*chi*p)*psi_left[:, 1]+np.exp(-1j*chi*p)*psi_right[:, 1]) + J_perp * psi[:, 0] - (U / 2)*(2*abs(psi[:, 1])**2)*psi[:, 1] + mu*psi[:, 1]
    return dydt.flatten() 

def GPE_condensate_with_mu(R,chi,U,p,L,mu,J_parallel =1,fluc_max = 0, phi =0,t0=0,step = 0.01,n_steps = 1000):
    time_start = time.time()
    J_perp = R
    energies = []
    final_states = []
    for i in range(0,L//2):
        omega1 = np.pi/4
        omega2 = np.pi/2
        omega1, cond1 = initial_cond_k(2*np.pi*i/L,omega1,fluc_max,R,chi, L,phi)
        omega2, cond2 = initial_cond_k(2*np.pi*i/L,omega2,fluc_max,R,chi, L,phi)
        omega,y0,final_state = compare_energy_with_mu(cond1,omega1,cond2,omega2,R,chi,U,L,mu,t0,step,n_steps)
        final_states = final_states + [[i,omega,final_state]]
        E =  Energy_functional_with_mu(final_state,J_parallel,J_perp,chi,U,L,mu)
        energies = energies + [E]
    best_state_index = np.argmin(energies)
    #print('GPE condensate computation took', time.time()-time_start)
    return final_states[best_state_index]

#mu_GPE =0
#e,f,g =GPE_condensate_with_mu(0.5,np.pi/2,0.1,1/2,20,mu_GPE,1,0,0,0,0.01,1000)
#norm_g = np.conj(g)@g
#print('GPE condensate and N WITH MU', np.abs(g), norm_g )
#mu_second_method = (Kinetic(g,1,0.5,np.pi/2) + 2* Interac(g,1,0.5,0.1))/norm_g 
#print('mu GPE  WITH MU',mu_second_method)

#################################################################################################################################################################
############################################################### BOGOLIUBOV #######################################################################################
##################################################################################################################################################################


a = 1
p = 1/2
J_parallel =1
U = 2
U=1
#U=0.01
U= 0.2
J_perp = 1.05
J_perp =0.87
J_perp = 0.89
J_perp = 0.99
J_perp = 1.5
J_perp = 0.0000001
J_perp =0.2
#J_perp = 1.05
R = J_perp/J_parallel
chi = np.pi/2
L= 24*4
L=200
L=100
L=4*8
L=5
#L= 4*10
#L=24*4
#L=24*4
#L=4
N=L
#N=1
regul = -0.0000001
regul =0

b,c,final_state=GPE_condensate(J_perp,chi,U,1/2,L,L,1,0,0,0,0.01,1) #   the condensate carries positive momentum if it's biased 
#final_state = psi_mix_k_new(-2*np.pi/L*b,R,chi/2,L,np.pi/2,0) #ok there is something about k and somehting about omega 
#final_state = np.array([0,0])
print('vortex or na and b',c,b)
psis_GS_R = final_state
#print('biased up and down', abs(final_state[0]),abs(final_state[1]))
#print('imbalance', np.abs(final_state))
#b=0
#c=0
#print('final state', final_state)
L = len(psis_GS_R)//2
N = round((np.conj((final_state))@final_state).real,0) 
#N =  round(np.conj((final_state)@final_state).real ,0)
#print('norm',  round((np.conj((final_state))@final_state).real,0) )
#print('L,N,R,U,chi,b,omega=', L,N,R,U,chi,b,c)
#print('b=',b)
#final_state = np.array([np.sqrt(N)* psi_mix_k(b*2*np.pi/L,j,m,R,chi,L,0,0) for j in range(L) for m in range(1,3)])
#pow = L//(2*b) if b!=0 else 1
pow =   L//math.gcd(2*b, L) if c == np.pi/4 else 1 
#print('pow', pow)
#pow = b
#pow = 10
#plot_psi(final_state,R,chi,U, L)
#print('GPE phase m =1', np.angle(final_state[0]*np.conj(final_state[2]))*L/(2*np.pi) ) #gives - momentum of the condensate
#print('GPE phase m = 2', np.angle(final_state[1]*np.conj(final_state[3]))*L/(2*np.pi) )
#print('b =', b)
#b=  np.angle(final_state[0]*np.conj(final_state[2]))*L/(2*np.pi)

def chemical_potential(psis,J_parallel,J_perp,chi,U):
    nb_bosons = round((np.conj(psis)@ psis).real,0)
    return (Kinetic(psis,J_parallel,J_perp,chi) + 2* Interac(psis,J_parallel,J_perp,U))/nb_bosons + regul

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

A1_R = A_m_Real( J_parallel,J_perp, U,chi, psis_GS_R,chi*(1-p),0)
A2_R = A_m_Real( J_parallel,J_perp, U, chi, psis_GS_R,-chi*p,1)
B1_R = B_m_Real(U, psis_GS_R,0)
B2_R = B_m_Real(U, psis_GS_R,1)
C1_R = C_m_Real(psis_GS_R, J_perp)
C2_R = C1_R
A1_R_hole = A_m_Real( J_parallel, J_perp, U, chi,psis_GS_R,chi*(1-p),0)
A2_R_hole = A_m_Real(J_parallel, J_perp,U,chi, psis_GS_R,-chi*p,1)

def construct_L(A1, B1, A2, B2, C1, C2,A1_hole,A2_hole, L,pow =1, epsilon= 0.00001):
    Translation = T_matrix_canonical_basis(pow,L)
    zero = np.zeros_like(A1)
    return np.block([[-A1, -C1, -B1, zero],[ -C2, -A2, zero, -B2], [B1.conj(), zero, A1_hole.conj(), C1.conj()],[zero, B2.conj(), C2.conj(), A2_hole.conj()] ]) + 2*epsilon*(Translation - np.conj(Translation).T)
    #return np.block([[-A1, -C1, -B1, zero],[ -C2, -A2, zero, -zero], [B1.conj(), zero, A1_hole.conj(), C1.conj()],[zero, zero, C2.conj(), A2_hole.conj()] ]) + 2*1j*epsilon*(Translation - np.conj(Translation).T)
    #return np.block([[A1, C1, B1, zero],[ C2, A2, zero, B2], [-B1.conj().T, zero, -A1_hole.conj().T, -C1.conj().T],[zero, -B2.conj().T, -C2.conj().T, -A2_hole.conj().T] ]) + 2*1j*epsilon*(Translation - Translation.T)




#psi1 =  np.reshape(psis_GS_R, (L,2))
#print('phase shift m=1', np.angle(psi1[4,0]*np.conj(psi1[5,0]))*L/(2*np.pi) )
#psi_test = np.array([ np.abs(psi1[0,0])*np.exp(1j*pow*b*2*np.pi/L*i) for i in range(L)  ] )

#psi1_leg1 = [psi1[j,0] for j in range(L)]

#TR = np.zeros((L, L), dtype=complex)
#for k in range(L):
#    TR[k,(k+pow)%L] = + 1*np.exp(-1j*2*pow*b*1*np.pi/L)

#print('commutator', np.max(np.abs(B1_R@ TR - TR @ B1_R )))


#B_test  = np.zeros((L, L), dtype=complex)
#for k in range(L):
    #B_test[k,k] = -U*psi_test[k]**2
    #B_test[k,k] = -U*psi1_leg1[k]**2
    #B_test[k,k] = -U*psi1[k,0]**2

#print('commutator test !!',   np.max(np.abs(B_test@ TR - TR @ B_test ))   )

#bogo spec
eom_matrix = construct_L(A1_R, B1_R, A2_R, B2_R, C1_R, C2_R,A1_R_hole,A2_R_hole,L,pow)
Translation = T_matrix_canonical_basis(pow,L)
print('commutator ladder',  np.max(np.abs(eom_matrix @ Translation - Translation @eom_matrix )))
eom_eigenvalues_unsorted, eom_eigenvectors_unsorted = np.linalg.eig(eom_matrix)
sorted_indices = np.argsort(eom_eigenvalues_unsorted)
eom_eigenvalues, eom_eigenvectors = eom_eigenvalues_unsorted[sorted_indices].real, eom_eigenvectors_unsorted[:, sorted_indices]
def symplectic_normalization_ladder(uv_eigenstate):
    u_values = uv_eigenstate[:2*L]
    v_values = np.conj(uv_eigenstate[2*L:])
    symp_norm = np.sqrt(np.sum(np.abs(u_values)**2) - np.sum(np.abs(v_values)**2) )
    #symp_norm = np.sqrt((np.abs(u_values)**2 - np.abs(v_values)**2 )*2*L)
    return u_values/symp_norm,v_values/symp_norm
def symplectic_normalization_ladder(uv_matrix): # 2L-1 columns and 4L rows 
    u_values = uv_matrix[:2*L,:]
    v_values = np.conj(uv_matrix[2*L:,:])
    #symp_norm = np.tile(np.sqrt(  np.sum( np.abs(u_values)**2) - np.sum(np.abs(v_values)**2, axis = 0)  ),(2*L,1))
    symp_norm = np.sqrt(  np.sum( np.abs(u_values)**2,axis = 0) - np.sum(np.abs(v_values)**2, axis = 0)  )
    print('output', np.sqrt(np.sum(np.abs( (u_values/symp_norm[np.newaxis, :])[:,0])**2) - np.sum(np.abs((v_values/symp_norm[np.newaxis, :])[:,0])**2) ) )
    return u_values/symp_norm[np.newaxis, :],  v_values/symp_norm[np.newaxis, :]

u_matrix,v_matrix = symplectic_normalization_ladder(eom_eigenvectors[:,2*L+1:])
ug, vg = u_matrix[:,0],v_matrix[:,0] #not the goldstone !!
print('R and U and chi and L', R,U,chi,L)
print('SYMPLECTIC NORM first mode',np.sqrt(np.sum(np.abs(ug)**2) - np.sum(np.abs(vg)**2) ) )

print('EOM eigenvalues zero mode and momentum ', eom_eigenvalues[2*L].real, np.angle(eom_eigenvectors[:,2*L][pow]*np.conj(eom_eigenvectors[:,2*L][0]))/pow)
#print('abs EOM unnorm eigenvectors zero mode and normalisation ', np.abs(eom_eigenvectors[:,2*L]), np.conj(eom_eigenvectors[:,2*L])@ eom_eigenvectors[:,2*L])
#print('for the zero mode no normalisation possible :( '  )

print('EOM eigenvalues first mode and momentum ', eom_eigenvalues[2*L+1].real, np.angle(eom_eigenvectors[:,2*L+1][pow]*np.conj(eom_eigenvectors[:,2*L+1][0]))/pow)
print('EOM eigenvectors first mode', np.abs(eom_eigenvectors[:,2*L+1]))
#print('check we get the same k with normalized eigenvect + abs eigenvect', np.angle(u_matrix[pow,0]* np.conj(u_matrix[0,0])), np.abs(u_matrix[:,0]),  np.abs(v_matrix[:,0]))

print('EOM eigenvalues second mode and momentum ', eom_eigenvalues[2*L+2].real,np.angle(eom_eigenvectors[:,2*L+2][pow]*np.conj(eom_eigenvectors[:,2*L+2][0]))/pow )
print('EOM eigenvectors second mode', np.abs(eom_eigenvectors[:,2*L+2]))
#print('check we get the same k with normalized eigenvect + abs eigenvect', np.angle(u_matrix[pow,1]* np.conj(u_matrix[0,1])), np.abs(u_matrix[:,1]),  np.abs(v_matrix[:,1]))

print('EOM eigenvalues third mode and momentum ', eom_eigenvalues[2*L+3].real,np.angle(eom_eigenvectors[:,2*L+3][pow]*np.conj(eom_eigenvectors[:,2*L+3][0]))/pow )
#print('EOM eigenvectors third mode', np.abs(eom_eigenvectors[:,2*L+3]))
#print('check we get the same k with normalized eigenvect + abs eigenvect', np.angle(u_matrix[pow,2]* np.conj(u_matrix[0,2])), np.abs(u_matrix[:,2]),  np.abs(v_matrix[:,2]))

print('EOM eigenvalues fourth mode and momentum ', eom_eigenvalues[2*L+4].real,np.angle(eom_eigenvectors[:,2*L+4][pow]*np.conj(eom_eigenvectors[:,2*L+4][0]))/pow )
#print('EOM eigenvectors fourth mode', np.abs(eom_eigenvectors[:,2*L+4]))
#print('check we get the same k with normalized eigenvect + abs eigenvect', np.angle(u_matrix[pow,3]* np.conj(u_matrix[0,3])), np.abs(u_matrix[:,3]),  np.abs(v_matrix[:,3]))



def gap_data(R_list, U_list, J_parallel=1, chi=np.pi/2, L=40, p=0.5, epsilon=1e-5):
    R_list = np.array(R_list)
    U_list = np.array(U_list)
    print('U list',U_list )
    gap_table = np.zeros((len(R_list), len(U_list)))
    for i, R in enumerate(R_list):
        for j, U in enumerate(U_list):
            J_perp = R * J_parallel
            #b, c, psi_GS = GPE_condensate(J_perp, chi, U, p, L, L, 1, 0, 0, 0, 0.01, 100)
            b,c,psi_GS=GPE_condensate(J_perp,chi,U,1/2,L,L,1,0,0,0,0.01,500)
            pow_val = L // np.gcd(2*b,L) if c == np.pi/4 else 1
            A1 = A_m_Real(J_parallel, J_perp, U, chi, psi_GS, chi * (1-p), 0)
            A2 = A_m_Real(J_parallel, J_perp, U, chi, psi_GS, -chi * p, 1)
            B1 = B_m_Real(U, psi_GS, 0)
            B2 = B_m_Real(U, psi_GS, 1)
            C1 = C_m_Real(psi_GS, J_perp)
            A1_hole = A1.copy()
            A2_hole = A2.copy()
            eom_matrix = construct_L(A1, B1, A2, B2, C1, C1, A1_hole, A2_hole, L, pow_val, epsilon)
            e_vals = np.sort(np.linalg.eigvals(eom_matrix).real)
            print('energy and U and R', e_vals[2*L+1],U, R )
            gap_table[i, j] = e_vals[2*L+1]
    return gap_table, R_list, U_list
def gap_data_vs_k(R_list, U_list, chi_list, L, J_parallel=1, p=0.5, epsilon=0):
    R_list = np.array(R_list)
    U_list = np.array(U_list)
    chi_list = np.array(chi_list)
    
    gap_table = np.zeros((len(R_list), len(U_list), len(chi_list)))
    k0_values = np.zeros_like(gap_table)

    for k, chi in enumerate(chi_list):
        for i, R in enumerate(R_list):
            for j, U in enumerate(U_list):
                J_perp = R * J_parallel
                b, c, psi_GS = GPE_condensate(J_perp, chi, U, p, L, L, 1, 0, 0, 0, 0.01, 300)
                k0 = 2 * np.pi * np.abs(b) / L
                k0_values[i, j, k] = k0

                pow_val = L // np.gcd(2 * b, L) if c == np.pi / 4 else 1
                A1 = A_m_Real(J_parallel, J_perp, U, chi, psi_GS, chi * (1 - p), 0)
                A2 = A_m_Real(J_parallel, J_perp, U, chi, psi_GS, -chi * p, 1)
                B1 = B_m_Real(U, psi_GS, 0)
                B2 = B_m_Real(U, psi_GS, 1)
                C1 = C_m_Real(psi_GS, J_perp)
                eom_matrix = construct_L(A1, B1, A2, B2, C1, C1, A1.copy(), A2.copy(), L, pow_val, epsilon)
                e_vals = np.sort(np.linalg.eigvals(eom_matrix).real)
                gap_table[i, j, k] = e_vals[2 * L + 1]

                print(f"Gap: {e_vals[2 * L + 1]:.5f}, k0: {k0:.5f}, R: {R}, U: {U}, chi: {chi:.5f}")

    return gap_table, k0_values, R_list, U_list, chi_list


def plot_gap_vs_R(gap_table, R_list, U_list, chi_index, chi_val, L):
    j = np.argmin(np.abs(U_list - U_list[0]))  # Assuming U is fixed
    plt.plot(R_list, gap_table[:, j, chi_index], 'o-')
    plt.xlabel('R', fontsize=30)
    plt.ylabel('Pseudo Goldstone gap', fontsize=30)
    plt.title(f'Gap vs R at χ = {chi_val:.2f}, U = {U_list[j]}, L = {L}', fontsize=30)
    plt.xticks(size=30)
    plt.yticks(size=30)
    plt.grid()
    plt.show()


def plot_gap_vs_U(gap_table, R_list, U_list, chi_index, chi_val, L):
    i = np.argmin(np.abs(R_list - R_list[0]))  # Assuming R is fixed
    plt.plot(U_list, gap_table[i, :, chi_index], 'o-')
    plt.xlabel('U', fontsize=30)
    plt.ylabel('Pseudo Goldstone gap', fontsize=30)
    plt.title(f'Gap vs U at χ = {chi_val:.2f}, R = {R_list[i]}, L = {L}', fontsize=30)
    plt.xticks(size=30)
    plt.yticks(size=30)
    plt.grid()
    plt.show()


def plot_gap_vs_chi_simple(gap_table, chi_list, R_fixed, U_fixed, R_list, U_list, L):
    i = np.argmin(np.abs(R_list - R_fixed))
    j = np.argmin(np.abs(U_list - U_fixed))
    y_vals = gap_table[i, j, :]
    
    plt.plot(chi_list, y_vals, 'o-')
    plt.xlabel(r'$\chi$', fontsize=30)
    plt.ylabel('Pseudo Goldstone gap', fontsize=30)
    plt.title(f'Gap vs χ at R = {R_fixed}, U = {U_fixed}, L = {L}', fontsize=30)
    plt.xticks(size=30)
    plt.yticks(size=30)
    plt.grid()
    plt.show()


#L = 96
#R_vals = np.linspace(0.005, 0.05, 5)
#R_vals = [0.1 ] 

#U_vals = np.linspace(0.005, 0.05, 5)
#U_vals = [1 ]
#chi_vals = np.linspace(0.6, np.pi * 0.99, 30)

#gap_table, k0_vals, R_arr, U_arr, chi_arr = gap_data_vs_k(R_vals, U_vals, chi_vals, L)

#plot_gap_vs_chi_simple(gap_table, chi_arr, R_fixed=0.1, U_fixed=1, R_list=R_arr, U_list=U_arr, L=L)

#chi_index = 10  # or any index within range of chi_vals
#plot_gap_vs_R(gap_table, R_arr, U_arr, chi_index, chi_vals[chi_index], L)
#plot_gap_vs_U(gap_table, R_arr, U_arr, chi_index, chi_vals[chi_index], L)


#plot_gap_vs_L_over_M(gap_table, k0_vals, R_arr, U_arr, R_fixed=0.01, U_fixed=0.01)


#R_vals = [0.1 ] 
#R_vals = np.linspace(0.001,0.1,15)

 
#U_vals = [1 ] 
#U_vals =  np.linspace(0.001,1,15) 
#gap_table, R_array, U_array = gap_data(R_vals, U_vals)
#plot_gap_vs_R(gap_table, R_array, U_array, U_fixed=1)
#plot_gap_vs_U(gap_table, R_array, U_array, R_fixed=0.1)









#for i in range(L):
#    print('Lk div 2pi', round(L/(2*np.pi)*np.angle(np.conj(eigenvectors_dyn[:,2*L+i][0])*eigenvectors_dyn[:,2*L+i][1]),2), round(L/(2*np.pi)*np.angle(np.conj(eigenvectors_dyn[:,2*L+i][2*L])*eigenvectors_dyn[:,2*L+i][2*L+1]),2) )

#print('tile', np.tile([0,1,2,3],(3,1)))

ug, vg = u_matrix[:,0],v_matrix[:,0] #not the goldstone !!
print('SYMPLECTIC NORM first mode',np.sqrt(np.sum(np.abs(ug)**2) - np.sum(np.abs(vg)**2) ) )
ug1, vg1 = u_matrix[:,1],v_matrix[:,1]
print('SYMPLECTIC NORM second mode',np.sqrt(np.sum(np.abs(ug1)**2) - np.sum(np.abs(vg1)**2) ) )
ug2, vg2 = u_matrix[:,2],v_matrix[:,2]
print('SYMPLECTIC NORM third mode',np.sqrt(np.sum(np.abs(ug2)**2) - np.sum(np.abs(vg2)**2) ))
ug3, vg3 = u_matrix[:,3],v_matrix[:,3]
ug4, vg4 = u_matrix[:,4],v_matrix[:,4]
print('norm test should be 0 first and second', np.sum(np.conj(ug1)*ug2 - np.conj(vg1)*vg2)  )
print('norm test should be 0  third and forth', np.sum(np.conj(ug3)*ug4 - np.conj(vg3)*vg4)  )
print(' phase for u and for v for second mode', np.angle( ug2[0]*np.conj(ug2[1])  ), np.angle( vg2[0]*np.conj(vg2[1])  )  )
print('i expect one for this norm in momentum space', (abs(ug2[0])**2 + abs(ug2[L])**2 - abs(vg2[0])**2 - abs(vg2[L])**2 )*L  ) 
columns = eom_eigenvectors[:,2*L+1:]
print('norm sum over bogo modes',np.sum( np.conj(columns[0,:])*columns[0,:]  -  np.conj(columns[2*L,:])*columns[2*L,:]    ) )

def is_it_diagonalizable(M,eigenvectors,eigenvalues):
    diagonalizable = True
    for i in range(len(eigenvalues)):
        Mv = M @ eigenvectors[:,i].T
        for j in range(len(eigenvalues)):
            if (eigenvalues[i]*eigenvectors[:,i][j] - Mv[j])> 0.00001:
                diagonalizable = False
    return diagonalizable



def group_degenerate_eigenvalues(eigenvalues, eps= 0.000001):
    groups = []
    distinct_eigenvalues = []  
    visited = np.zeros(len(eigenvalues), dtype=bool) 
    #print('eigenvalues',eigenvalues)
    for i, eigval in enumerate(eigenvalues):
        #print(visited[i])
        if visited[i]:
            continue
        degenerate_set = np.where(np.abs(eigenvalues - eigval) < eps)[0]
        #print('abs diff', np.abs(eigenvalues - eigval) )
        #print('arg', np.abs(eigenvalues - eigval) < eps)
        #print('where', np.where(np.abs(eigenvalues - eigval) < eps))
        #print('dege set', degenerate_set)
        visited[degenerate_set] = True
        groups.append(degenerate_set)
        distinct_eigenvalues.append(eigval.real) 
    return groups, distinct_eigenvalues


def new_eigenbasis(L_matrix_eigvectors, T_matrix, groups):
    """Outputs the eigenbasis that diagonalizes T and L"""
    new_eigenvectors = np.copy(L_matrix_eigvectors) 
    for group in groups:
        degenerate_subspace = L_matrix_eigvectors[:, group] 
        T_submatrix = degenerate_subspace.T.conj() @ T_matrix @ degenerate_subspace 
        sub_eigenvalues, U = np.linalg.eig(T_submatrix) 
        new_eigenvectors[:, group] = degenerate_subspace @ U
    return new_eigenvectors



def eigenvector_profile(eigenvector, L):
    eigenvector_reshaped = np.reshape(eigenvector, (4, L))  
    density_1 = np.abs(eigenvector_reshaped[1, :])
    phase_1 = np.angle(eigenvector_reshaped[1, :])
    density_2 = np.abs(eigenvector_reshaped[3, :])
    phase_2 = np.angle(eigenvector_reshaped[3, :])
    #print('eigenvector !!!',eigenvector)
    #print('eig reshaped',np.reshape(eigenvector, (4, L))  )
    #print('here2', eigenvector_reshaped[:, 1])
    #print('here!', phase_1)
    
    x = np.arange(L)
    #plt.plot(x, 10*density_1, 'r', label='Density 1',linestyle ='-')
    #plt.plot(x, 10*density_2, 'b', label='Density 2',linestyle ='-')
    plt.plot(x, phase_1, 'g', label='phase 1', linestyle ='-')
    plt.plot(x, phase_2, 'y', label='phase 2', linestyle ='-')
    #plt.plot(x, phase_1, 'b', label='Phase 1')
    plt.legend()
    plt.show()

#l = [i for i in range(1,41)]
#l = np.reshape(l, (10, 4)) 
#print(l)
#print(l[1,:])
#eigenvector = new_eigenvectors[:, 1]
#eigenvector = ivan_new_eigenvectors[:, np.argmin(np.abs(np.real(ivan_eigenvalues)))]
#eigenvector = ivan_new_eigenvectors[:, np.argmin(np.abs(np.real(ivan_eigenvalues)))]
#print('eigenvalues',ivan_eigenvalues)
#print('eigenvector !!', eigenvector)
#print(ivan_new_eigenvalues[0])
#eigenvector_profile(eigenvector,L)


def Bogoliubov_spectrum_real_space_plot(eigenvalues,eigenvectors,chi,R,U): #I want to make sure i plot moemnta relative to the condensate
    k_values = []
    energy_values = []
    energy_before_SSB_up = []
    energy_before_SSB_down = []
    for i in range(4*L):
        
        k = np.angle(eigenvectors[:,i][0]*np.conj(eigenvectors[:,i][pow]))/pow + b*2*np.pi/L if c==0 else np.angle(eigenvectors[:,i][0]*np.conj(eigenvectors[:,i][pow]))/pow # -k_0 +q   + k_0
        #k = np.angle(eigenvectors[:,i][0]*np.conj(eigenvectors[:,i][pow]))/pow - b*2*np.pi/L 
        trans = T_matrix_canonical_basis(pow,L)
        k= np.angle((np.conj(eigenvectors[:,i]) @ trans)[0]*eigenvectors[:,i][0])/pow
        
        print('k from T, and real k', k,np.angle((np.conj(eigenvectors[:,i]) @ trans)[0]*eigenvectors[:,i][0]) )
        #k = np.angle(eigenvectors[:,i][0]*np.conj(eigenvectors[:,i][pow]))/pow 
        #k_prime = np.angle(eigenvectors[:,i][2*L]*np.conj(eigenvectors[:,i][2*L +1]))/1
        #print(' L*k/(2pi), should be integer',round(L*k/(2*np.pi),2), round(L*k_prime/(2*np.pi),2) )
        #print(round(k,2),round(eigenvalues[i].real,2),eigenvector)
        #k_values.append(k *L/(2*np.pi) )
        k_values.append(k/(np.pi) ) 
        energy_values.append(eigenvalues[i].real)
        energy_before_SSB_up.append(spec(R,1,k,chi) - spec(R,1,2*b*np.pi/L,chi))
        energy_before_SSB_down.append(spec(R,2,k,chi) - spec(R,1,2*b*np.pi/L,chi))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    plt.plot(k_values,energy_values , marker='o', linestyle='None',markersize = 5,lw=30, color='#DC143C')
    #plt.plot(k_values,energy_before_SSB_up , marker='o', linestyle='None',markersize = 5,lw=30, color='green')
    #plt.plot(k_values,energy_before_SSB_down , marker='o', linestyle='None',markersize = 5,lw=30, color='blue')
    plt.grid()
    #plt.title( r' $\chi =$'+f' {round(chi,2)}, R = {R}, U = {U}, L={L}',size = 30, pad=20)
    #plt.title( r' $\chi =\frac{\pi}{2},$'+f' R = {R}, U = {U}, L={L}',size = 30, pad=20)
    plt.title( r' $\chi =\frac{\pi}{2},$'+f' R = {R}, U = {U}, L={L}',size = 30, pad=20)
    plt.xlabel(r'\textit{momentum k (units of $\pi$)}',size = 30)
    plt.xticks(size = 30)
    plt.yticks(size = 30)
    plt.ylabel(r'\textit{Bogoliubov energy}',size = 30)
    #ylabel(r'\textit{voltage (mV)}',fontsize=16)
    #plt.xlim(-0.5, 0.5)  
    plt.ylim(-0.8, 0.8)  
    plt.show()

#Bogoliubov_spectrum_real_space_plot(eom_eigenvalues, eom_eigenvectors,chi,R,U)

def Bogo_spec_real_space(eigenvalues, u_eigenvectors, L):
    k_values = {}  
    for i in range(len(eigenvalues)):
        k = np.angle(u_eigenvectors[:,i][0]*np.conj(u_eigenvectors[:,i][pow]))/pow + b*2*np.pi/L if c==0 else np.angle(u_eigenvectors[:,i][0]*np.conj(u_eigenvectors[:,i][pow]))/pow # -k_0 +q   + k_0
        #print('k !!!', k)
        k_rounded = round(k, 5)  
        if k_rounded not in k_values:
            k_values[k_rounded] = [] 
        k_values[k_rounded].append((i, eigenvalues[i].real))
    return k_values

#bogo_eigenvectors_unnormalized = eom_eigenvectors[:,2*L+1:]
#symp_norm = np.sqrt((np.abs(bogo_eigenvectors_unnormalized[:2*L,:])**2 - np.abs(bogo_eigenvectors_unnormalized[2*L:,:])**2)*2*L)
#print("sym norm", symp_norm)
#bogo_ueigenvectors = bogo_eigenvectors_unnormalized[:2*L,:] / symp_norm 
#bogo_veigenvectors = np.conj(bogo_eigenvectors_unnormalized)[2*L:,:] / symp_norm 
#bogo_eigenvalues = eom_eigenvalues[2*L+1:]

#print('bogo group',Bogo_spec_real_space(bogo_eigenvalues,bogo_ueigenvectors, L))
#k_values = Bogo_spec_real_space(bogo_eigenvalues, bogo_ueigenvectors, L)

def plot_bogo_leg_imbalance(k_values, u_eigenvectors, L):
    lower_branch_diff = []
    upper_branch_diff = []
    k_list = []
    for k in sorted(k_values.keys()):
        modes = sorted(k_values[k], key=lambda x: x[1])  # Sort modes by energy
        half = len(modes) // 2
        lower_branch = modes[:half] 
        upper_branch = modes[half:]  
        lower_diff = []
        upper_diff = []
        for idx, _ in lower_branch:
            lower_diff.append(abs(u_eigenvectors[:, idx][0])**2 - abs(u_eigenvectors[:, idx][L])**2)
        for idx, _ in upper_branch:
            upper_diff.append(abs(u_eigenvectors[:, idx][0])**2 - abs(u_eigenvectors[:, idx][L])**2)
        lower_branch_diff.append(np.mean(lower_diff))
        upper_branch_diff.append(np.mean(upper_diff))
        k_list.append(k / np.pi)

    plt.figure(figsize=(8, 6))
    plt.plot(k_list, lower_branch_diff, label="Lower branch imbalance", marker='o', color='b')
    plt.plot(k_list, upper_branch_diff, label="Upper branch imbalance", marker='x', color='r')
    plt.xlabel('Momentum (k/π)')
    plt.ylabel('Population Imbalance')
    plt.title('Population Imbalance vs. Momentum')
    plt.legend()
    plt.grid(True)
    plt.show()

    return k_list, lower_branch_diff, upper_branch_diff
#plot_bogo_leg_imbalance(k_values,bogo_ueigenvectors, L)


def calculate_imbalance_and_momentum(u_values, v_values, L):
    imbalance_u = np.zeros(u_values.shape[1])  
    imbalance_v = np.zeros(v_values.shape[1]) 
    k_values = np.zeros(u_values.shape[1]) 

    for i in range(u_values.shape[1]):

        #imbalance_u[i] = (np.abs(u_values[0, i])**2 - np.abs(u_values[L, i])**2) / (np.abs(u_values[0, i])**2 + np.abs(u_values[L, i])**2)
        #imbalance_u[i] = (np.abs(u_values[0, i])**2 - np.abs(u_values[1, i])**2)
        imbalance_u[i] = (np.abs(u_values[0, i])**2 + np.abs(v_values[0, i])**2) / (np.abs(u_values[L, i])**2 + np.abs(v_values[L, i])**2)

        imbalance_v[i] = (np.abs(v_values[0, i])**2 - np.abs(v_values[L, i])**2) / (np.abs(v_values[0, i])**2 + np.abs(v_values[L, i])**2)
        #imbalance_v[i] = (np.abs(v_values[0, i])**2 - np.abs(v_values[1, i])**2) 

        k_values[i] = -np.angle(u_values[0, i] * np.conj(u_values[1, i])) 
        #print('kval', k_values[i]*L/(2*np.pi) )

    return imbalance_u, imbalance_v, k_values

# Function to plot leg imbalance for each mode classified by momentum
def plot_imbalance_by_momentum(uv_matrix, L, N):

    u_normalized, v_normalized = symplectic_normalization_ladder(uv_matrix)
    imbalance_u, imbalance_v, k_values = calculate_imbalance_and_momentum(u_normalized, v_normalized, L)
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(k_values, imbalance_u, marker='o', linestyle='None', markersize=5, lw=2, color='b')
    plt.xlabel('Momentum k',fontsize =35)
    #plt.ylabel(r'$\frac{|u^1| - |u^2| }{|u^1| + |u^2|}$',fontsize =35)
    plt.ylabel(r'$\frac{|u^1| + |v^1| }{|u^2| + |v^2|}$',fontsize =35)
    plt.title(f'U={U}, R={R}, L={L}, N={100},'+ r' $\chi = \frac{\pi}{2}$', fontsize = 35)
    #plt.title('Leg Imbalance for u-values (Bogoliubov modes) by Momentum')
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(k_values, imbalance_v, marker='o', linestyle='None', markersize=5, lw=2, color='r')
    plt.xlabel('Momentum k',fontsize =35)
    plt.ylabel(r'$\frac{|v^1| - |v^2| }{|v^1| + |v^2|}$',fontsize =35)
    #plt.title('Leg Imbalance for v-values (Bogoliubov modes) by Momentum')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#u_matrix,v_matrix = symplectic_normalization_ladder(eom_eigenvectors[:,2*L+1:])
#uv_matrix = np.random.random((4 * L, N))  # Replace with actual uv_matrix

# Call the function to plot the imbalance by momentum
#plot_imbalance_by_momentum(eom_eigenvectors[:,2*L+1:], L, N)


####################################################################################################################################################################
############################################################# Bogo in momentum space Meissner exact ################################################
####################################################################################################################################################################


#U_values = [0.0, 0.05, 0.1, 0.5, 1, 1.5]
#q_vals = np.linspace(-np.pi, np.pi, 200) 


#n, R, chi, L = 0.5, 1, np.pi/2, 200 
#Emin = np.min(spec(R, 1, 0, chi))

#def chem_pot(U, n, R, chi):
#    k0 = k_minus(R, chi, L)
#    theta_val = theta(k0, R, chi)
#    return spec(R, 1, 0, chi) + U * n 


def A(q, U, n, R, chi, L, mu):
    return spec(R,0,q,chi)+2*U*n-mu
def B(q, U, n, R, chi, L, mu):
    return U*n*np.sin(2*theta(q,R,chi))
def C(q, U, n, R, chi, L, mu):
    return U*n*np.cos(2*theta(q,R,chi))

def D(q, U, n, R, chi, L, mu):
    return spec(R,1,q,chi)+2*U*n-mu

def a_tilde(q, U, n, R, chi, L, mu):
    return -A(q,U,n,R,chi,L,mu)**2 + 2*B(q,U,n,R,chi,L,mu)**2 + 8*C(q,U,n,R,chi,L,mu)**2 - D(q,U,n,R,chi,L,mu)**2

def c_tilde(q, U, n, R, chi, L, mu):
    A_val = A(q,U,n,R,chi,L,mu)
    B_val = B(q,U,n,R,chi,L,mu)
    C_val = C(q,U,n,R,chi,L,mu)
    D_val = D(q,U,n,R,chi,L,mu)
    return (B_val**4 + 16*C_val**4
            - 8*A_val*C_val**2*D_val
            + B_val**2*(8*C_val**2 - D_val**2)
            + A_val**2*(-B_val**2 + D_val**2))

def omega_plus(q, U, n, R, chi, L, mu):
    a = a_tilde(q,U,n,R,chi,L,mu)
    c = c_tilde(q,U,n,R,chi,L,mu)
    return (-a + np.sqrt(a**2 - 4*c)) / 2

def omega_minus(q, U, n, R, chi, L, mu):
    a = a_tilde(q,U,n,R,chi,L,mu)
    c = c_tilde(q,U,n,R,chi,L,mu)
    return (-a - np.sqrt(a**2 - 4*c)) / 2

def omega_minus_approx(q, U, n, R, chi, L, mu):
    D_val = spec(R, 1, q, chi) + 2*U*n - mu
    B_val = U*n*np.sin(2 * theta(q, R, chi))
    return np.sqrt(D_val**2 - B_val**2)

#fig, axes = plt.subplots(2, 3, figsize=(18, 10))
#axes = axes.flatten()

#for i, U_val in enumerate(U_values):
#    mu_val = chem_pot(U_val, n, R, chi)

#    omega_plus_vals = [np.sqrt(omega_plus(q, U_val, n, R, chi, L, mu_val)) for q in q_vals]
#    omega_minus_vals = [np.sqrt(omega_minus(q, U_val, n, R, chi, L, mu_val)) for q in q_vals]
#    omega_approx_vals = [omega_minus_approx(q, U_val, n, R, chi, L, mu_val) for q in q_vals]
    
#    free_plus = [spec(R, 0, q, chi) - mu_val for q in q_vals]
#    free_minus = [spec(R, 1, q, chi) - mu_val for q in q_vals]

#    ax = axes[i]
#    #ax.plot(q_vals, omega_plus_vals, label=r'$\omega_+$', color='b', lw=2)
#    ax.plot(q_vals, omega_minus_vals, label=r'$\omega_-$', color='r', lw=2)
#    ax.plot(q_vals, omega_approx_vals, label=r'$\omega_{approx}$', linestyle='--', lw=2)

#    #ax.plot(q_vals, free_plus, color='b', lw=2, alpha=0.3, label='free +')
#    #ax.plot(q_vals, free_minus, color='r', lw=2, alpha=0.3, label='free -')

#    ax.set_title(f'U = {U_val}', fontsize=14)
#    ax.set_xlabel(r'$k_0 + q$', fontsize=12)
#    ax.set_ylabel('E', fontsize=12)
#    ax.grid(True)
#    ax.legend()
#    #ax.set_xlim(-1, 1)
#    #ax.set_ylim(-0.2, 0.2)


#plt.tight_layout()
#plt.show()

# omega_minus vs omega_approx 
#plt.figure(figsize=(10, 6))
#U_values = [0.0, 0.05, 0.1, 0.5, 1, 2, 3]

#for U_val in U_values:
#    mu_val = chem_pot(U_val, n, R, chi)
#    omega_m = [np.sqrt(omega_minus(q, U_val, n, R, chi, L, mu_val)) for q in q_vals]
#    omega_a = [omega_minus_approx(q, U_val, n, R, chi, L, mu_val) for q in q_vals]

#    plt.plot(q_vals, omega_m, label=rf'$\omega_-$ (U={U_val})')
#    plt.plot(q_vals, omega_a, '--', label=rf'$\omega_{{approx}}$ (U={U_val})')

#plt.xlabel(r'$k_0 + q$', fontsize=16)
#plt.ylabel('E', fontsize=16)
#plt.title('Spectrum for Different $U$', fontsize=18)
#plt.grid(True)
#plt.legend(fontsize=12)
#plt.tight_layout()
#plt.show()

####################################################################################################################################################################
############################################################# Bogo in momentum space + Chevy for 1D Bose gas and 3D ################################################
####################################################################################################################################################################

m_I = 1/2
m_B = 1/2  #we assume mass of bosons = mass of impurity -2cos(k)+ 2 \sim k^2 so m = 1/2 ??
m_red = m_I*m_B/(m_I +m_B)
L=5
n = 0.5
#R = 1
U=0.1
#chi = np.pi/2
g = -1
multiple = 6
multiple = 1
cutoff = multiple*2*np.pi
Gamma = 0.25
#dk =  0.1
dk = 2*np.pi/L

V = L**3
k_values_3D = [dk*i for i in range(int(round(cutoff/dk,0)))]

def chem_potential(U,n):
    return -2 + U*n
    #return U*n

def free_spec(k,mu):
    return -2*np.cos(k*a  ) -mu
    #return k**2/(2*m_B) - mu   

def spec_impurity(k):
    return -2*np.cos(k*a ) +2
    #return k**2/(2*m_I) 

def reduced_spec(m1,m2,k,U,n):
    mu = chem_potential(U,n)
    m_red = m1*m2/(m1 +m2)
    #return k**2/(2*m_red)
    return free_spec(k,mu) + spec_impurity(k) 


def S(k):
    return 4*np.pi*k**2

def inv_g_strength(k_range,t,puls,m1,m2,L):
    #polarization_bubble = sum( dk*S(k_range[id])/((2*np.pi)**3)/(puls - reduced_spec(m1,m2,k_range[id],R,U,n) -U*n) for id in range(1,len(k_range)) )
    polarization_bubble = 1/L*sum(1/(puls - reduced_spec(m1,m2,k_range[id],U,n) +U*n  ) for id in range(1,len(k_range)) )
    return 1/t  + polarization_bubble
#inv_t_values = np.linspace(-10,10,50)
#inv_g_values = [inv_g_strength(k_values_3D,1/t,0,m_I,m_B,L)  for t in inv_t_values]
#plt.plot(inv_t_values,inv_g_values)
#plt.show()
def bogo_spec(k,U,n):
    Ek = free_spec(k ,chem_potential(U,n))
    return np.sqrt(abs((Ek + U*n)*(Ek +3*U*n)))

def eta(k,U,n):
    if k==0 :
        return 0
    else :
        Ek = free_spec(k,chem_potential(U,n))
        return 1/4*np.log(abs((Ek + U*n)/(Ek + 3*U*n)))


def uk_eta(k,U,n):
    return np.cosh(eta(k,U,n))
def vk_eta(k,U,n):
    return np.sinh(eta(k,U,n))

def Wk(k,U,n):
    Ek = free_spec(k,chem_potential(U,n))
    epsk = bogo_spec(k,U,n)
    uk =uk_eta(k,U,n)
    vk =vk_eta(k,U,n)
    #return np.sqrt(abs((Ek + U*n)/(epsk)))
    return uk +vk
    

def uk_Wk(k,U,n):
    W =Wk(k,U,n)
    return 1/2*(W +1/W)

def vk_Wk(k,U,n):
    W =Wk(k,U,n)
    return 1/2*(W -1/W)


def V1kkprime(k,k_prime,U,n):
    uk =uk_eta(k,U,n)
    vk =vk_eta(k,U,n)
    uk_prime =uk_eta(k_prime,U,n)
    vk_prime =vk_eta(k_prime,U,n)
    W =Wk(k,U,n)
    W_prime =Wk(k_prime,U,n)
    #return 1/2*(W*W_prime + 1/(W*W_prime) )
    return uk*uk_prime + vk*vk_prime

def lippmann_Schwinger_g(k_range,E,U,n,L):
    #summation = sum (  dk*S(k_range[id])/((2*np.pi)**3)/(E - free_spec(k_range[id],chem_potential(U,n)) - spec_impurity(k_range[id]) -U*n )  for id in range(1,len(k_range)))
    #summation = 1/L*sum (  1/(E - free_spec(k_range[id],chem_potential(U,n)) - spec_impurity(k_range[id]) )  for id in range(1,len(k_range)))
    summation = 1/L*sum (  1/(E - reduced_spec(m_I,m_B,k_range[id],U,n) - U*n )  for id in range(1,len(k_range)))
    return 1/(summation)



#x = np.linspace(0,np.pi,1000)
#y1=  Wk(x,U,n)
#y2 = V1kkprime(x,x,U,n)
#plt.plot(x,y1,color='r')
#plt.plot(x,y2,color='g')
#plt.show()



def Frolich_Hamiltonian(U,n,g,L):
    " I organise the psi{p,k} in an array psi_{1,0} psi_{1,1} .... psi_{1,L} psi_{2,0} psi_{2,1} .... psi_{2,L}..... "
    H_Frolich = np.zeros((L*(L),L*(L)),dtype = np.complex64)
    cste = g/L*sum( vk_eta(2*np.pi/L*k_1,U,n)**2  for k_1 in range(1,L))
    print("cste!!",cste)
    for p in range(L):
        for k in range(L):
            for k_prime in range(L):
                V = V1kkprime(2*np.pi/L *k ,2*np.pi/L *k_prime,U,n) if (k!= 0 and k_prime!=0) else 0
                impurity_kin =   -2*np.cos(2*np.pi/L*(p-k)) +2  if (k==k_prime ) else 0
                #impurity_kin =   (2*np.pi/L*(p-k))**2/(2*m_I)  if (k==k_prime) else 0
                bath = bogo_spec(2*np.pi/L*k,U,n)  if (k==k_prime and k!= 0) else 0
                BI_int_cste1 = g*n if (k==k_prime) else 0
                BI_int_cste2 = cste if (k==k_prime) else 0
                BI_int_W = 0
                if k == 0 and k_prime != 0:
                    BI_int_W = g*np.sqrt(n/L)*Wk(2*np.pi/L*k_prime, U, n)
                elif k != 0 and k_prime == 0:
                    BI_int_W= g*np.sqrt(n/L)*Wk(2*np.pi/L*k,  U, n)
                BI_int_V =  g/L*V 
                #BI_int_cste2 = 0
                #BI_int_cste1 = 0
                #BI_int_V = 0
                #BI_int_W = 0
                H_Frolich[p*(L) + k,p*(L) + k_prime] =  impurity_kin + bath + BI_int_cste1 + BI_int_cste2 + BI_int_W + BI_int_V
    return H_Frolich

def Frolich_Hamiltonian_new(p,U,n,g,L,multiple =1):
    " I organise the psi{p,k} in an array psi_{1,0} psi_{1,1} .... psi_{1,L} psi_{2,0} psi_{2,1} .... psi_{2,L}..... "
    H_Frolich = np.zeros((multiple*L,multiple*L),dtype = np.complex64)
    cste = g/L*sum( vk_eta(2*np.pi/L*k_1,U,n)**2  for k_1 in range(1,multiple*L))
    print('constant',cste)
    #cste = g*dk/((2*np.pi)**3)*sum(vk_eta(2*np.pi/L*kid,U,n)**2*S(2*np.pi/L*kid) for kid in range(1,multiple*L))
    for k in range(multiple*L):
        for k_prime in range(multiple*L):
            V = V1kkprime(2*np.pi/L *k ,2*np.pi/L *k_prime,U,n) if (k!= 0 and k_prime != 0) else 0
            BI_int_V =  g/L*V 
            #BI_int_V = g*dk/((2*np.pi)**3)*V1kkprime(2*np.pi/L *k,2*np.pi/L *k_prime,U,n)* np.sqrt(S(2*np.pi/L *k) * S(2*np.pi/L *k_prime)) if (k!= 0 and k_prime != 0) else 0
            #impurity_kin =   -2*np.cos(2*np.pi/L*(L/(2*np.pi)*p-k)) +2  if (k==k_prime ) else 0
            #impurity_kin =   (2*np.pi/L*(L/(2*np.pi)*p-k))**2/(2*m_I)  if (k==k_prime ) else 0
            impurity_kin =   spec_impurity(  (2*np.pi/L*(L/(2*np.pi)*p-k)) ) if (k==k_prime ) else 0
            bath = bogo_spec(2*np.pi/L*k,U,n)  if ( k==k_prime and k!= 0) else 0
            BI_int_cste1 = g*n if ( k==k_prime) else 0
            BI_int_cste2 = cste if (k==k_prime ) else 0
            BI_int_W = 0
            if k == 0 and k_prime != 0:
                BI_int_W = g*np.sqrt(n/L)*Wk(2*np.pi/L*k_prime, U, n)
                #BI_int_W = g*np.sqrt(n)*Wk(2*np.pi/L*k_prime,U,n)* np.sqrt(S(2*np.pi/L*k_prime)*dk/((2*np.pi)**3))
            elif k != 0 and k_prime == 0:
                BI_int_W= g*np.sqrt(n/L)*Wk(2*np.pi/L*k, U, n)
                #BI_int_W = g*np.sqrt(n)*Wk(2*np.pi/L*k,U,n)* np.sqrt(S(2*np.pi/L*k)*dk/((2*np.pi)**3))
            #BI_int_W = 0
            #BI_int_V = 0
            #BI_int_cste2 = 0
            #bath = 0
            #impurity_kin = 0
            H_Frolich[ k, k_prime] =  impurity_kin + bath + BI_int_cste1 + BI_int_cste2 + BI_int_W + BI_int_V
    return H_Frolich

def Hamiltonian_2Boglons_approx(p,U,n,g,L,multiple=1):
    "I want to organise my psi^p_{k_1,k_2} where the couple (k_1,k_2) can be whatever except (0, non zero) : (k_1,k_2) = (0,0) (1,0) (1,1) (1,2)....(1,L-1) (2,0) (2,1)... (2,L-1) (3,0)... "
    H_matrice = np.zeros( (  1 + L*(L-1) , 1 + L*(L-1)), dtype = complex)
    cste = g/L*sum( vk_eta(2*np.pi/L*k_1,U,n)**2  for k_1 in range(1,multiple*L)) + g*n
    print('constant',cste)
    for index in range( L*(L-1)):
        k_1,k_2 = (0,0) if ( index ==0) else  (index//L  +1,  (index-1)%L)
        for index_prime in range(  L*(L-1)):
            k_1_prime,k_2_prime = (0,0) if ( index_prime ==0) else  (index_prime//L  +1,  (index_prime-1)%L )
            same_index = index == index_prime
            same_anti_index = (k_1 == k_2_prime and k_2 == k_1_prime)
            impurity_kin =   spec_impurity(  2*np.pi/L*(L/(2*np.pi)*p-k_1-k_2)   ) if (index==index_prime ) else 0
            bath = 0
            if index != 0 and index_prime != 0 and ( same_index or same_anti_index  ):
                bath =  bogo_spec(2*np.pi/L*k_1,U,n) if (k_2 == 0 and k_2_prime == 0) else bogo_spec(2*np.pi/L*k_1,U,n) + bogo_spec(2*np.pi/L*k_2,U,n)

            BI_cste = cste if (same_index or same_anti_index) else 0
            BI_V = g/L* V1kkprime(2*np.pi/L*k_1_prime ,2*np.pi/L *k_1,U,n) if (index!= 0 and index_prime != 0 and ( k_2_prime == k_2 or k_2_prime == k_1 or k_1_prime == k_2 or k_1_prime == k_1) ) else 0
            BI_W = 0
            if k_2 == k_2_prime and k_2 ==0:
                if  k_1 == 0 and k_1_prime != 0:
                    BI_W += g*np.sqrt(n/L)*Wk(2*np.pi/L*k_1_prime, U, n) 
                elif k_1 != 0 and k_1_prime == 0:
                    BI_W += g*np.sqrt(n/L)*Wk(2*np.pi/L*k_1, U, n)
            elif  k_2 == 0 and k_2_prime != 0:
                if k_1 == k_2_prime :
                    BI_W += g*np.sqrt(n/L)*Wk(2*np.pi/L*k_1_prime, U, n) 
                if k_1 == k_1_prime :
                    BI_W += g*np.sqrt(n/L)*Wk(2*np.pi/L*k_2_prime, U, n)
            elif k_2 != 0 and k_2_prime == 0:
                if k_1_prime == k_2:
                    BI_W += g*np.sqrt(n/L)*Wk(2*np.pi/L*k_1, U, n)
                if k_1_prime == k_1 :
                    BI_W += g*np.sqrt(n/L)*Wk(2*np.pi/L*k_2, U, n)
            #BI_cste =0
            #BI_W =0
            #BI_V =0
            #bath =0
            #impurity_kin =0

            H_matrice[index,index_prime] = BI_cste + BI_W + BI_V +bath + impurity_kin

    return H_matrice
H_mat = Hamiltonian_2Boglons_approx(2*np.pi/L,1,0.5,1,L)
print('Hemiticity check', np.max( H_mat - np.conj(H_mat).T))



def Frolich_Hamiltonian_new_gmode(p,U,n,g,L,multiple =1):
    " I organise the psi{p,k} in an array psi_{1,0} psi_{1,1} .... psi_{1,L} psi_{2,0} psi_{2,1} .... psi_{2,L}..... "
    H_Frolich = np.zeros((multiple*(L+1),multiple*(L+1)),dtype = np.complex64)
    cste = g/L*sum( vk_eta(2*np.pi/L*k_1,U,n)**2  for k_1 in range(multiple*L))
    #cste = g*dk/((2*np.pi)**3)*sum(vk_eta(2*np.pi/L*kid,U,n)**2*S(2*np.pi/L*kid) for kid in range(1,multiple*L))
    for k in range(multiple*(L+1)):
        for k_prime in range(multiple*(L+1)):
            V = V1kkprime(2*np.pi/L *(k%L) ,2*np.pi/L *(k_prime%L),U,n) if (k!= 0 and k_prime != 0) else 0
            BI_int_V =  g/L*V 
            #BI_int_V = g*dk/((2*np.pi)**3)*V1kkprime(2*np.pi/L *k,2*np.pi/L *k_prime,U,n)* np.sqrt(S(2*np.pi/L *k) * S(2*np.pi/L *k_prime)) if (k!= 0 and k_prime != 0) else 0
            impurity_kin =   spec_impurity(  (2*np.pi/L*(L/(2*np.pi)*p-k%L))) if (k==k_prime ) else 0
            bath = bogo_spec(2*np.pi/L*(k%L),U,n)  if ( k==k_prime and k!= 0) else 0
            BI_int_cste1 = g*n if ( k==k_prime) else 0
            BI_int_cste2 = cste if (k==k_prime ) else 0
            BI_int_W = 0
            if k == 0 and k_prime != 0:
                BI_int_W = g*np.sqrt(n/L)*Wk(2*np.pi/L*(k_prime%L), U, n)
                #BI_int_W = g*np.sqrt(n)*Wk(2*np.pi/L*k_prime,U,n)* np.sqrt(S(2*np.pi/L*k_prime)*dk/((2*np.pi)**3))
            elif k != 0 and k_prime == 0:
                BI_int_W= g*np.sqrt(n/L)*Wk(2*np.pi/L*(k%L), U, n)
                #BI_int_W = g*np.sqrt(n)*Wk(2*np.pi/L*k,U,n)* np.sqrt(S(2*np.pi/L*k)*dk/((2*np.pi)**3))
            #impurity_kin =  0
            #bath = 0
            #BI_int_W = 0
            #BI_int_V = 0
            #BI_int_cste2 = 0
            H_Frolich[ k, k_prime] =  impurity_kin + bath + BI_int_cste1 + BI_int_cste2 + BI_int_W + BI_int_V
    return H_Frolich

#mat_test = Frolich_Hamiltonian_new_gmode(0,U,n,g,L,1)
#print('hermitician ??', np.max(np.abs(Frolich_Hamiltonian_new_gmode(0,U,n,g,L,1) - np.conj(Frolich_Hamiltonian_new_gmode(0,U,n,g,L,1)).T )))

def Frolich_Hamiltonian_3D(p, U, n, g,L): # p=0
    H_Frolich_3D = np.zeros( (len(k_values_3D), len(k_values_3D)) , dtype=complex)
    for i, k1 in enumerate(k_values_3D):
        for j, k2 in enumerate(k_values_3D):
            if i == j:
                sum_term = g*dk/((2*np.pi)**3)*sum(vk_eta(k_values_3D[kid],U,n)**2*S(k_values_3D[kid]) for kid in range(1,len(k_values_3D)))
                interaction_term_V =g*dk/((2*np.pi)**3)*V1kkprime(k1,k2,U,n)* S(k1) if i!=0 else 0
                H_Frolich_3D[i,j] = spec_impurity(k1)  +bogo_spec(k1,U,n) + g*n + sum_term + interaction_term_V
                #H_Frolich_3D[i,j] = free_spec(k1,chem_potential(U,n)) +bogo_spec(k1,U,n) + g*n + sum_term
            elif i ==0 and j!= 0:
                interaction_term_W = g*np.sqrt(n)*Wk(k2,U,n)* np.sqrt(S(k2)*dk/((2*np.pi)**3))
                #interaction_term_W =g*np.sqrt(n)*Wk(k2,U,n)* S(k2)
                H_Frolich_3D[i,j] = interaction_term_W
                #print('W term', H_Frolich_3D[i,j])
            elif j ==0 and i!= 0:
                interaction_term_W = g*np.sqrt(n)*Wk(k1,U,n)* np.sqrt(S(k1)*dk/((2*np.pi)**3))
                #interaction_term_W = g*np.sqrt(n)*Wk(k1,U,n)* S(k1)
                H_Frolich_3D[i,j] = interaction_term_W
                #print('W term 2', H_Frolich_3D[i,j])
            #elif j!=0 and i !=0 and i != j:
            else:
                interaction_term_V = g*dk/((2*np.pi)**3)*V1kkprime(k1,k2,U,n)* np.sqrt(S(k1) * S(k2))
                #interaction_term_V = g/L*V1kkprime(k1,k2,U,n)*S(k1)* S(k2)
                H_Frolich_3D[i,j] = interaction_term_V
                #print('V term', H_Frolich_3D[i,j])
                #print('kk matrix element',interaction_term_V)
    return H_Frolich_3D

#Frolich_3D1 = Frolich_Hamiltonian_3D(0, U, n, g, L)
#Frolich_3D2 = Frolich_Hamiltonian_new(0,U,n,g,L,multiple )
#print('length k vAL 3d', len(k_values_3D))
#print('length MAT 1', multiple*L)
#print("diff mat1 mat2 frolich", np.max(np.abs(Frolich_3D1  - Frolich_3D2 )))
#print('Froch 3D matrix', Frolich_3D)
#print(' hermiticity frolich  3D!!',np.max(np.abs(Frolich_3D - np.conj(Frolich_3D).T)))

#eigenvalues_Frolich_3D, eigenvectors_Frolich_3D = np.linalg.eigh(Frolich_3D)
#print('3D chevy eigen values and vectors',eigenvalues_Frolich_3D, eigenvectors_Frolich_3D)
def psi_k(nb,Q,U,n,g,L):
    matrix_H = Frolich_Hamiltonian_new(Q,U,n,g,L)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix_H)
    #print('eigenvalues', eigenvalues)
    #print('chevy',eigenvalues, eigenvectors)
    #psi_ks = [abs(eigenvectors[0][i+1])**2 for i in range(len(eigenvectors) -1)]
    print('shape of eigen', eigenvectors.shape)
    print('len eigen range', range(len(eigenvectors) ))
    print('len eigen', len(eigenvectors))
    print('eigen', eigenvectors[0][199], eigenvectors[0][0] )
    print('list', [i for i in range(len(eigenvectors) )])
    psi_ks = [abs(eigenvectors[0][i+1])**2 for i in range(len(eigenvectors) -1 )]
    #psi_ks = [abs(eigenvectors[nb][i])**2 for i in range(len(eigenvectors)  )]

    #print(psi_ks)
    ks = [2*np.pi/L*i for i in range(1,L)]
    #ks = [2*np.pi/L*i for i in range(L)]
    plt.plot(ks,psi_ks)
    plt.ylabel(r"$|\psi_k|^2$", fontsize = 30)
    plt.xlabel(r'k', fontsize = 30)
    plt.xticks(size = 30)
    plt.yticks(size = 30)
    #plt.yscale('log')
    plt.title(f'{nb}-th excited state in chevy subspace, Q = {Q}, U={U},n={n}, g={g}, L={L}', fontsize = 30)
    plt.show()
#psi_k(199,0,U,n,g,200)
time_start = time.time()

#Frolich = Frolich_Hamiltonian(U,n,g,L)
#Frolich = Frolich_Hamiltonian_new(U,n,g,L)
#print('frolich ',Frolich)
#print("diff",np.max(np.abs(Frolich - Frolich.conj().T)))

#eigenvalues,eigenvectors = np.linalg.eigh(Frolich)


#eigenvalues_Frolich, eigenvectors_Frolich = np.linalg.eigh(Frolich)
#print('FROLICH EIGENVECTORS FOR W=0', eigenvectors_Frolich)
#print(eigenvalues_Frolich, eigenvectors_Frolich)

eps = 0.25
def spectral_function(Q,Omega,eigenvalues,eigenvectors):
    summation = 0
    for i in range(len(eigenvalues)):
        oscillator_strength = np.abs(eigenvectors[:,i][ int(L/(2*np.pi)*Q*(L)) ])**2  #psi_p basically
        summation +=oscillator_strength/(Omega - eigenvalues[i]+ 1j*eps)
    return -2*summation.imag
#The old one is faster
#Q_vals = [2*np.pi/L*i -np.pi for i in range(L) ]
Omega = 0
#y_vals = [spectral_function(Q,Omega,eigenvalues_Frolich,eigenvectors_Frolich) for Q in Q_vals]
#plt.scatter(Q_vals,y_vals)
#plt.xlabel(' impurity momentum Q')
#plt.ylabel(' spectral function')
#plt.title(r'$\Omega = 0$'+ f' g={g}')
#print('This took', time.time()-time_start, 'seconds')
#plt.show()

def spectral_function_new(Q,Omega,U,n,g,L):
    #Froch = Frolich_Hamiltonian_new(Q,U,n,g,L)
    Froch = Frolich_Hamiltonian_3D(Q, U, n, g, L)
    eigenvalues, eigenvectors = np.linalg.eigh(Froch)
    summation = 0
    for i in range(len(eigenvalues)):
        oscillator_strength = np.abs(eigenvectors[:,i][ 0 ])**2  #psi_p basically
        summation += oscillator_strength/(Omega - eigenvalues[i]   + 1j*eps)
    return -2*summation.imag

def spectral_function_new(Q,Omega,eigenvalues,eigenvectors):
    #Froch = Frolich_Hamiltonian_new(Q,U,n,g,L)
    #Froch = Frolich_Hamiltonian_3D(Q, U, n, g, L)
    #eigenvalues, eigenvectors = np.linalg.eigh(Froch)
    summation = 0
    for i in range(len(eigenvalues)):
        oscillator_strength = np.abs(eigenvectors[:,i][ 0 ])**2  #psi_p basically
        summation += oscillator_strength/(Omega - eigenvalues[i]   + 1j*eps)
    return -2*summation.imag

def spectral_function_table(Q_vals, Omega_vals, U, n, g, L, eps, Frolich_Hamiltonian_new):
    t_start = time.time()
    num_Q = len(Q_vals)
    num_Omega = len(Omega_vals)
    eigvals_list = []
    eigvecs_list = []

    for Q in Q_vals:
        H = Frolich_Hamiltonian_new(Q, U, n, g, L)
        eigvals, eigvecs = np.linalg.eigh(H)
        eigvals_list.append(eigvals)
        eigvecs_list.append(eigvecs)

    eigvals_array = np.array(eigvals_list)      
    eigvecs_array = np.stack(eigvecs_list, axis=0)  

    oscillator_strengths = np.abs(eigvecs_array[:, 0, :]) ** 2  

    Omega_vals = Omega_vals[:, None, None] 
    eigvals_array = eigvals_array[None, :, :]  
    osc = oscillator_strengths[None, :, :]   

    denom = Omega_vals - eigvals_array + 1j * eps
    spectral_contributions = osc / denom
    spectral_vals = -2 * np.sum(spectral_contributions.imag, axis=2)  

    print(f"Full spectral function computed in {time.time() - t_start:.2f} seconds.")
    return spectral_vals

#print('spectral fct for a given Q and omega (old)', spectral_function(2*np.pi/L*1,Omega,eigenvalues,eigenvectors))
#print('spectral fct for a given Q and omega (new)', spectral_function_new(2*np.pi/L*1,Omega,U,n,g,L))

Q_vals = np.array([2*np.pi/L*i  -np.pi for i in range(multiple*L) ])
Omega = 0
#y_vals = [spectral_function_new(Q,Omega,U,n,g,L) for Q in Q_vals]
#plt.scatter(Q_vals,y_vals)
#plt.xlabel(' impurity momentum Q')
#plt.ylabel(' spectral function')
#plt.title(r'$\Omega = 0$'+ f' g={g}')
#print('This took', time.time()-time_start, 'seconds')
#plt.show()
Q=0
#Omega_vals =np.linspace(-10,10,400)
#y_vals = [spectral_function_new(Q,Omega,U,n,g,L) for Omega in Omega_vals]
#plt.scatter(Omega_vals,y_vals)
#plt.xlabel(' Omega')
#plt.ylabel(' spectral function')
#plt.title(r'$\Omega = 0$'+ f' g={g}')
#print('This took', time.time()-time_start, 'seconds')
#plt.show()

Omega_vals = np.linspace(-10,10,5*L)



def compute_rho_0_spectral_function(Q_vals, Omega_vals, U, n, gamma, L):
    k_vals = np.arange(L) * (2 * np.pi / L)  
    q_grid, k_grid = np.meshgrid(Q_vals, k_vals, indexing='ij')  
    shifted_k = (q_grid - k_grid)  
    omega_I = spec_impurity(shifted_k) 
    omega_B = bogo_spec(k_vals, U, n)[None, :]
    total_energy = omega_B + omega_I + g*n + g/L*sum( vk_eta(2*np.pi/L*k_1,U,n)**2  for k_1 in range(1,multiple*L))
    max_energy_Q = np.max(total_energy, axis=1) 
    min_energy_Q = np.min(total_energy, axis=1)  
    Omega_vals_expanded = Omega_vals[:, None, None]  
    energy_diff = Omega_vals_expanded - total_energy[None, :, :] 
    lorentzian = gamma / (np.pi * (energy_diff ** 2 + gamma ** 2))
    spectral_values = np.sum(lorentzian, axis=2) 
    return spectral_values, max_energy_Q, min_energy_Q

def plot_rho_0_spectral_function(Q_vals, Omega_vals, spectral_vals):
    Q_mesh, Omega_mesh = np.meshgrid(Q_vals, Omega_vals, indexing='ij')
    norm = mcolors.LogNorm(vmin=max(spectral_vals.min(), 1e-6), vmax=spectral_vals.max())

    plt.figure(figsize=(8, 6))
    plt.scatter(Q_mesh, Omega_mesh, c=spectral_vals.T, cmap='viridis', norm=norm, s=40)
    plt.colorbar(label=r"$\rho_0(\Omega,Q)$")  # Adjust font size for colorbar
    plt.xlabel("Impurity momentum $Q$", fontsize=14)
    plt.ylabel(r"$\Omega$", fontsize=14)
    plt.title(r" $\rho_0(\Omega,Q)$ for " + f'U={U}, n={n}, L={L}', fontsize=16)
    plt.tight_layout()
    plt.show()

#spectral_vals, max_energy_Q, min_energy_Q = compute_rho_0_spectral_function(Q_vals, Omega_vals, U, n, Gamma, L)
#plot_rho_0_spectral_function(Q_vals, Omega_vals, spectral_vals)


def spectral_function_plot(f, Q_vals,Omega_vals,eigenvalues,eigenvectors,g):
    spectral_values = np.zeros( (  (len(Omega_vals)), len(Q_vals) ) )
    Omega_mesh , Q_mesh= np.meshgrid( Omega_vals,Q_vals, indexing='ij')   
    timee = time.time()
    for i in range(len(Omega_vals)):
        spectral_values[i,:] = np.array([ f(Q, Omega_vals[i], eigenvalues, eigenvectors ) for Q in Q_vals ])
        print(f'completed omega = {Omega_vals[i]} in', time.time()-timee, 'seconds !!')
    #norm = mcolors.Normalize(vmin=spectral_values.min(), vmax=spectral_values.max())
    norm = mcolors.LogNorm(vmin=max(spectral_values.min(), 1e-6), vmax=spectral_values.max())
    plt.scatter(Q_mesh, Omega_mesh, c=spectral_values, cmap='viridis', norm=norm, s=35)
    plt.colorbar(label=r"$\mathcal{A}(\Omega,Q)$")
    plt.xlabel("impurity momentum Q")
    plt.ylabel(r"$\Omega$")
    plt.title(f'g = {g},' + f' broadening =' + f'{eps}, L={L}, U= {U},n = {n}' )
    plt.show()


def spectral_function_plot_new(f, Q_vals,Omega_vals,U,n,g,L):
    Q_mesh, Omega_mesh = np.meshgrid(Q_vals, Omega_vals, indexing='ij')  
    times = time.time()
    spectral_values = np.array([ [f(Q, Omega, eigvals, eigvecs)  for Omega in Omega_vals ] for Q, (eigvals, eigvecs) in zip(Q_vals, [np.linalg.eigh(Frolich_Hamiltonian_new(Q, U, n, g, L, multiple)) for Q in Q_vals])])
    
    print('spectral function vs omega vs Q took', time.time() - times, ' seconds')

    norm = mcolors.LogNorm(vmin=max(spectral_values.min(), 1e-6), vmax=spectral_values.max())
    plt.scatter(Q_mesh, Omega_mesh, c=spectral_values, cmap='viridis', norm=norm, s=35)
    plt.colorbar(label=r"$\mathcal{A}(\Omega,Q)$")
    plt.xlabel("impurity momentum Q")
    plt.ylabel(r"$\Omega$")
    plt.title(f'g = {g},' + f' broadening =' + f'{eps}, L={L}, U= {U},n = {n}'  )
    plt.show()

def spectral_function_plot_table(Q_vals, Omega_vals, spectral_vals, g, L, U, n, eps):
    Q_mesh, Omega_mesh = np.meshgrid(Q_vals, Omega_vals, indexing='ij')
    norm = mcolors.LogNorm(vmin=max(spectral_vals.min(), 1e-6), vmax=spectral_vals.max())

    plt.figure(figsize=(8, 6))
    plt.scatter(Q_mesh, Omega_mesh, c=spectral_vals.T, cmap='viridis', norm=norm, s=40)
    #plt.scatter(Q_vals, max_energy_Q , s=5, color ='red')
    #plt.scatter(Q_vals, min_energy_Q , s=5, color ='red')
    cbar = plt.colorbar(label=r"$\mathcal{A}(\Omega,Q)$")
    cbar.set_label(r"$\mathcal{A}(\Omega,Q)$", fontsize=15) 
    
    plt.xlabel("Impurity momentum $Q$",size =15)
    plt.ylabel(r"$\Omega$",size = 15)
    plt.xticks(size=15)
    plt.yticks(size=15)
    plt.title(f"$g={g}$, $\Gamma={eps}$, $L={L}$, $U={U}$, $n={n}$", size = 15)
    plt.tight_layout()
    plt.show()


##spec_vals = spectral_function_table(Q_vals, Omega_vals, U, n, g, L, eps, Frolich_Hamiltonian_new)
##spectral_function_plot_table(Q_vals, Omega_vals, spec_vals, g, L, U, n, eps)









#timee = time.time()
#the old one is faster
#spectral_function_plot(spectral_function, Q_vals,Omega_vals,eigenvalues_Frolich,eigenvectors_Frolich,g)
#print('spectral fct plot Q omega old took', time.time()- timee)
#Note that this can't make sense for 3D case because we do not have acess to multiple Q cb isotropy --> Q=0


#timee = time.time()
print('about to call')
#spectral_function_plot_new(spectral_function_new, Q_vals,Omega_vals,U,n,g,L)
#print('spectral fct plot Q omega new took', time.time()- timee)

def plot_spectral_function_plot_omega_g(f,g_vals,Omega_vals,Q,U,n,L):
    spectral_values = np.zeros(((len(Omega_vals)), len(g_vals)))
    Omega_mesh , g_mesh  = np.meshgrid( Omega_vals, g_vals,indexing='ij')  
    time_s= time.time()
    for i in range(len(g_vals)):
        Frolich_H = Frolich_Hamiltonian(U,n,g_vals[i],L)
        eigenvalues,eigenvectors = np.linalg.eigh(Frolich_H)
        spectral_values[:, i] = np.array([f(Q, x, eigenvalues, eigenvectors )
                                 for x in Omega_vals] )
        print('g =', 1/g_vals[i], " it took", time.time()-time_s,' seconds')
    norm = mcolors.Normalize(vmin=spectral_values.min(), vmax=spectral_values.max())
    plt.scatter( g_mesh,Omega_mesh, c=spectral_values, cmap='viridis', norm=norm, s=35)
    plt.colorbar(label=r"$\mathcal{A}(\Omega,Q)$")
    plt.title(f'Q={Q}, broadening =' + f'{eps}, L={L}, U= {U},n = {n}' )
    plt.ylabel(r"$\Omega$")
    plt.xlabel('g')
    plt.show()

m_red = m_I*m_B/(m_I +m_B)
#kn = (6*np.pi**2*n)**(1/3)
kn = 1
#En = kn**2/(4*m_red)
En= 1

def plot_spectral_function_plot_omega_g_new(f,inv_ak_vals,Omega_vals,Q,U,n,L): 

    #inv_t_vals = inv_ak_vals/( 2*np.pi/m_red/kn )
    #alt t_vals
    #inv_t_vals = inv_ak_vals*( 2*np.pi/m_red/kn )
    inv_t_vals = inv_ak_vals

    #inv_g_vals = [inv_g_strength(k_values_3D ,1/el,0,m_I,m_B,L) for el in inv_t_vals]
    #inv_g_vals = inv_t_vals 
    #alt g_vals
    #inv_g_vals = [1/inv_g_strength(k_values_3D ,el,0,m_I,m_B,L) for el in inv_t_vals]
    inv_g_vals = inv_t_vals 

    #inv_g_Lippmann_Schwinger = [1/lippmann_Schwinger_g(k_values_3D,om,U,n,L) for om in Omega_vals[:len(Omega_vals)//2]]
    #alt
    inv_g_Lippmann_Schwinger = [lippmann_Schwinger_g(k_values_3D,om,U,n,L) for om in Omega_vals[:len(Omega_vals)//2]]
    
    #polariz_bubble = -sum(dk*S(k_values_3D[id])/((2*np.pi)**3)/(reduced_spec(m_I,m_B,k_values_3D[id],U,n) +U*n) for id in range(1,len(k_values_3D)) )
    #polariz_bubble = -1/L*sum(1/(reduced_spec(m_I,m_B,k_values_3D[id],U,n)  )for id in range(1,len(k_values_3D)) )

    #inv_t_Lippmann_Schwinger = [ inv_g - polariz_bubble for inv_g in inv_g_Lippmann_Schwinger]
    #inv_t_Lippmann_Schwinger = inv_g_Lippmann_Schwinger
    #alt
    #inv_t_Lippmann_Schwinger = [ 1/(1/inv_g - polariz_bubble) for inv_g in inv_g_Lippmann_Schwinger]
    inv_t_Lippmann_Schwinger = inv_g_Lippmann_Schwinger

    #inv_ak_Lippmann_Schwinger =[el * ( 2*np.pi/m_red/kn )  for el in inv_t_Lippmann_Schwinger ]
    #alt
    #inv_ak_Lippmann_Schwinger =[el / ( 2*np.pi/m_red/kn )  for el in inv_t_Lippmann_Schwinger ]
    inv_ak_Lippmann_Schwinger =[el  for el in inv_t_Lippmann_Schwinger ]


    spectral_values = np.zeros( (  (len(Omega_vals)), len(inv_ak_vals) ) )
    Omega_mesh , inv_ak_mesh  = np.meshgrid( Omega_vals, inv_ak_vals,indexing='ij')  
    time_s= time.time()
 
    for i in range(len(inv_g_vals)):
        #Froch = Frolich_Hamiltonian_3D(Q, U, n, 1/inv_g_vals[i], L)
        #Froch = Frolich_Hamiltonian_new(Q,,U,n,1/inv_g_vals[i],L,multiple)
        #alt
        Froch = Frolich_Hamiltonian_new(Q,U,n,inv_g_vals[i],L,multiple)
        
        eigenvalues, eigenvectors = np.linalg.eigh(Froch)

        #spectral_values[:, i] = np.array([f(Q,x,U,n,1/inv_g_vals[i],L )
        #                        for x in Omega_vals] )
        spectral_values[:, i] = np.array([f(Q,x,eigenvalues,eigenvectors )
                                for x in Omega_vals] )
        print('1/(ka) =', inv_ak_vals[i], " it took", time.time()-time_s,' seconds')
    #norm = mcolors.Normalize(vmin=spectral_values.min(), vmax=spectral_values.max())
    norm = mcolors.LogNorm(vmin=max(spectral_values.min(), 1e-6), vmax=spectral_values.max())
    plt.plot(inv_ak_Lippmann_Schwinger,Omega_vals[:len(Omega_vals)//2],linestyle='-',color='red',lw=2)
    plt.xlim(min(inv_ak_vals),max(inv_ak_vals))
    plt.ylim(min(Omega_vals),max(Omega_vals))
    plt.scatter( inv_ak_mesh,Omega_mesh, c=spectral_values, cmap='viridis', norm=norm, s=35)
    plt.colorbar(label=r"$\mathcal{A}(\Omega,Q)$")
    plt.title(f'Q={Q}, broadening =' + f'{eps}, L={L}, U= {U},n = {n}' )
    plt.ylabel(r"$\Omega$")
    #plt.xlabel(r'$(a_{BI})^{-1}$')
    #plt.xlabel(r'$a_{BI}$')
    #plt.xlabel(r'$1/g$')
    plt.xlabel(r'$g$')
    plt.show()


Q=0
#inv_ak_vals = np.linspace(-3,3,50)*(6*np.pi**2*n)**(1/3)
#alt
#inv_ak_vals = np.linspace(-5,5,100)/(6*np.pi**2*n)**(1/3)
#Omega_vals = np.linspace(-5,5,100)*((6*np.pi**2*n)**(1/3))**2/(4*m_red)
inv_ak_vals = np.linspace(-30,30,100)
Omega_vals = np.linspace(-30,30,100)
timee = time.time()
#plot_spectral_function_plot_omega_g(spectral_function,g_vals,Omega_vals,Q,U,n,L)
#print("old version took", time.time()- timee)
timee = time.time()
#new is faster
#plot_spectral_function_plot_omega_g_new(spectral_function_new,g_vals,Omega_vals,Q,U,n,L)
#print("new version took", time.time()- timee)

#3D
#plot_spectral_function_plot_omega_g_new(spectral_function_new,inv_ak_vals,Omega_vals,Q,U,n,L)
#print("new version took", time.time()- timee)

#####################################################################################################################################################################################
################################################################ Bogo in real space + Chevy for 1D Bose gas + dynamical structure factor #########################################################################
#####################################################################################################################################################################################

#hmm start with Bogoliubov
flux =8*2*np.pi/50*0
n = 0.5
U= 0.1
L= 500
L= 40
L=2
N = int(L*n)
g = -5
epsilon = 0.00000
regularisation = 0.0000001
regularisation =0
Gamma = 0.05
mu = (-2 + U*n)*1  
condensate = np.array([np.sqrt(n)*np.exp(+1j*flux*i) for i in range(L)])


def bogo_spec(k,U,n):
    Ek = free_spec(k-flux ,chem_potential(U,n))
    return np.sqrt(abs((Ek + U*n)*(Ek +3*U*n)))


def Aji(condensate,U):
    off =0.00000
    mu = -2 + U*n - regularisation
    mat = np.zeros((L,L),dtype = complex)
    for i in range(L):
        mat[i,(i+1)%L] = 1*np.exp(-1j*flux) + 1j*off
        mat[i,(i-1)%L] = 1*np.exp(1j*flux)  - 1j*off
        mat[i,i] = -2*U*abs(condensate[i])**2 + mu
    return mat

def Bji(condensate,U):
    mat = np.zeros((L,L),dtype = complex)
    for i in range(L):
        mat[i,i] = -U*condensate[i]**2
    return mat

def T_matrix_1D(pow,L):
    Tr = np.zeros((L, L), dtype=complex)
    zero = np.zeros_like(Tr)
    for j in range(L):
        #Tr[j,(j+pow)%L] = + np.exp(-1j*flux*pow*2)
        Tr[j,(j+pow)%L] = 1
    return np.block([[np.exp(-1j*flux)*Tr, zero ],[zero ,np.exp(1j*flux)* Tr] ])

def eom(condensate, U,pow,L,epsilon = 0.00000000001):
    Translation = T_matrix_1D(pow,L)
    A = Aji(condensate,U)
    B = Bji(condensate,U)
    return np.block([[-A, -B ],[np.conj(B), np.conj(A) ] ]) +1*epsilon*(Translation  - np.conj(Translation).T )


#bogo spec
dyn_mat =eom(condensate, U,1,L)
eigenvalues_dyn_unsorted ,eigenvectors_dyn_unsorted= np.linalg.eig(dyn_mat)
sorted_indices = np.argsort(eigenvalues_dyn_unsorted.real)

#eigenvalues_dyn, eigenvectors_dyn = eigenvalues_dyn_unsorted[sorted_indices][L+1:].real, eigenvectors_dyn_unsorted[:, sorted_indices][:,L+1:]
eigenvalues_dyn, eigenvectors_dyn = eigenvalues_dyn_unsorted[sorted_indices].real, eigenvectors_dyn_unsorted[:, sorted_indices]
Bogo_spec, Bogo_eigenvectors = eigenvalues_dyn_unsorted[sorted_indices][L+1:].real, eigenvectors_dyn_unsorted[:, sorted_indices][:,L+1:]

#for i in range(L):
#    print('Lk div 2pi', round(L/(2*np.pi)*np.angle(np.conj(eigenvectors_dyn[:,L+i][0])*eigenvectors_dyn[:,L+i][1]),2), round(L/(2*np.pi)*np.angle(np.conj(eigenvectors_dyn[:,L+i][L])*eigenvectors_dyn[:,L+i][L+1]),2) )
    #print(' eigenvalue', eigenvalues_dyn[i])
#    print('eig val and vector and norm', eigenvalues_dyn[i+L], eigenvectors_dyn[:,i+L ], np.sum( np.conj(eigenvectors_dyn[:,i+L ])@ eigenvectors_dyn[:,i+L ]) )
#    print('symplectic norm',  np.sum( np.conj(eigenvectors_dyn[:L,i+L ]) @ eigenvectors_dyn[:L,i+L ]) - np.sum( np.conj(eigenvectors_dyn[L:,i+L ]) @ eigenvectors_dyn[L:,i+L ])   )


#T_matr = T_matrix_1D(1,L)
#Tr = np.zeros((L, L), dtype=complex)
#for j in range(L):
#    Tr[j,(j+pow)%L] = + np.exp(-1j*flux)
    #Tr[j,(j-pow)%L] = - np.exp(-1j*flux)

#print('commutator 1D', np.max(np.abs(Tr @ Bji(condensate,U)  - Bji(condensate,U)  @ Tr )))

#print('commutator 1D', np.max(np.abs(T_matr @ dyn_mat  - dyn_mat  @ T_matr )))



#indices, eigenvals = group_degenerate_eigenvalues(eigenvalues_dyn,0.000000001)
#print('group dege, ', indices, eigenvals)
#eigenvectors_dyn_new = new_eigenbasis(eigenvectors_dyn, T_matr, indices)

#for i in range(L):
#    print('NEW Lk div 2pi', round(L/(2*np.pi)*np.angle(np.conj(eigenvectors_dyn_new[:,L+i][0])*eigenvectors_dyn_new[:,L+i][1]),2), round(L/(2*np.pi)*np.angle(np.conj(eigenvectors_dyn_new[:,L+i][L])*eigenvectors_dyn_new[:,L+i][L+1]),2) )
    #print(' eigenvalue', eigenvalues_dyn[i])
#    print('eig val and vector and norm', eigenvalues_dyn[i+L], eigenvectors_dyn[:,i+L ], np.sum( np.conj(eigenvectors_dyn[:,i+L ])@ eigenvectors_dyn[:,i+L ]) )
#    print('symplectic norm',  np.sum( np.conj(eigenvectors_dyn[:L,i+L ]) @ eigenvectors_dyn[:L,i+L ]) - np.sum( np.conj(eigenvectors_dyn[L:,i+L ]) @ eigenvectors_dyn[L:,i+L ])   )

def bogo_plot(eigenvalues_dyn ,eigenvectors_dyn):
    k_vals = []
    k_prime_vals = []
    spec = []
    spec_prime = []
    spec_exact = []
    spec_exact_prime = []
    for i in range(len(eigenvalues_dyn)):
        k = (np.angle(eigenvectors_dyn[:,i][0]* np.conj(eigenvectors_dyn[:,i][1])) +flux) # -flux -q + flux
        k_prime = (np.angle(eigenvectors_dyn[:,i][L]* np.conj(eigenvectors_dyn[:,i][L+1])) -flux) #  flux -q - flux
        #print('u [:,L+1:]shift v shift', round(np.angle(eigenvectors_dyn[:,i][0]* np.conj(eigenvectors_dyn[:,i][1]) )*L/(2*np.pi) +flux,2), round(np.angle(eigenvectors_dyn[:,i][L]* np.conj(eigenvectors_dyn[:,i][L+1]))*L/(2*np.pi) -flux,2) )
        #print('k and k prime', k*L/(2*np.pi),k_prime *L/(2*np.pi))
        #print('k == k_prime ?', round(k*L/(2*np.pi)-k_prime *L/(2*np.pi),1)  %L ==0 )
        #k= np.angle(np.conj(eigenvectors_dyn[:,i]).T @T_matr @ eigenvectors_dyn[:,i])
        #print('flux =',round(flux*L/(2*np.pi),2))
        #print('k=',k*L/(2*np.pi))
        #print('k=',k*L/(2*np.pi))
        #print('k+flux, bogo spec',round((k+flux)*L/(2*np.pi),2) ,bogo_spec(k+flux ,U,n))
        k_vals.append(k)
        k_prime_vals.append(k_prime)
        spec.append(eigenvalues_dyn[i])
        spec_prime.append(eigenvalues_dyn[i])
        #spec_exact.append(bogo_spec(k+flux ,U,n))
        spec_exact_prime.append(bogo_spec(k_prime+flux ,U,n))

    plt.scatter(k_vals,spec)
    #plt.scatter(k_prime_vals,spec_prime, c= 'g')
    #plt.scatter(k_vals,spec_exact, c ='r')
    #plt.scatter(k_prime_vals,spec_exact_prime, c ='y')
    plt.show()
#bogo_plot(eigenvalues_dyn ,eigenvectors_dyn)

#for i in range(len(eigenvalues_dyn)):
#    print('eigval and eigvec',eigenvalues_dyn[i], eigenvectors_dyn[:,i])
#print(" the same eigenvector ??", eigenvectors_dyn[:,0] -eigenvectors_dyn[:,1])
#print('eig values and vectors, ', eigenvalues_dyn,eigenvectors_dyn)
#print('group degen eigenvalues',group_degenerate_eigenvalues(eigenvalues_dyn.real, 0.01) )

def symplectic_normalization_1D(uv_matrix,L): # L-1 columns and 2L rows 
    u_values = uv_matrix[:L,:]
    v_values = np.conj(uv_matrix[L:,:])
    symp_norm = np.sqrt(  np.sum( np.abs(u_values)**2,axis = 0) - np.sum(np.abs(v_values)**2, axis = 0)  )
    new_u, new_v = u_values/symp_norm[np.newaxis, :],  v_values/symp_norm[np.newaxis, :]
    print('normalisation !!', np.sqrt( np.sum(np.abs(new_u)**2,axis = 0) - np.sum(np.abs(new_v)**2,axis =0) ))
    return new_u, new_v 

def symplectic_normalization_1D(uv_eigenstate,L):  # L-1 columns and 2L rows 
    u_values = uv_eigenstate[:L,:]
    v_values = np.conj(uv_eigenstate[L:,:])
    symp_norm = np.sqrt( np.sum(np.abs(u_values)**2,axis = 0) - np.sum(np.abs(v_values)**2,axis =0) )
    new_u, new_v = u_values/symp_norm,v_values/symp_norm
    print('normalisation !!', np.sqrt( np.sum(np.abs(new_u)**2,axis = 0) - np.sum(np.abs(new_v)**2,axis =0) ))
    return new_u, new_v 
print('check point')

symplectic_normalization_1D(Bogo_eigenvectors,L)

def dyn_structure_factor_1D(q_vals,Omega_vals,Bogo_spec,condensate,Bogo_eigenvectors,Gamma = 0.25):
    L = len(condensate)
    phi_star_j = np.conj(condensate)[:,None,None,None]
    phi_j = condensate[:,None,None,None]
    u_unnormalized = Bogo_eigenvectors[:L,:]
    v_unnormalized = np.conj(Bogo_eigenvectors[L:,:])  
    symplectic_norm = np.sqrt((np.abs(u_unnormalized)**2 - np.abs(v_unnormalized)**2)*L)
    u,v = u_unnormalized/ symplectic_norm, v_unnormalized/symplectic_norm
    #print("goldstone mode !!", u[:,0])
    #print('symplectic norm', (np.abs(u)**2-np.abs(v)**2)*L )
    vj_mu = v[:,:,None,None]  #index j mu q omega
    u_star_j_mu = np.conj(u)[:,:,None,None] 
    phase_factor =np.exp(1j*q_vals[None,None,:,None]*np.arange(L)[:,None,None,None])
    denominator = (Omega_vals[None,None,:] -Bogo_spec[:,None,None] + 1j*Gamma)
    super_oscillator_strength = phase_factor*(  phi_star_j * vj_mu + phi_j * u_star_j_mu )
    oscillator_strength = np.abs(np.sum(super_oscillator_strength ,axis = (0) ))**2
    spec_matrix = np.sum( oscillator_strength/denominator,axis =0 )
    return -2/L**2*spec_matrix.imag

q_vals = 2*np.pi/L*np.arange(L) -np.pi 
Omega_vals = np.linspace(-10,10,2*L)
#Omega_vals = np.linspace(-2,7,2*L)
print('len omega, q', len(Omega_vals), len(q_vals))
#dyn_structure_factor_1D(q_vals,Omega_vals,eigenvalues_dyn,condensate,eigenvectors_dyn,Gamma )

def dyn_structure_factor_1D_plot(f, q_vals, Omega_vals, Bogo_spec, condensate, Bogo_eigenvectors, L,Gamma=0.25):
    spectral_values = f(q_vals, Omega_vals, Bogo_spec, condensate, Bogo_eigenvectors, Gamma)
    Q_mesh, Omega_mesh = np.meshgrid(q_vals, Omega_vals, indexing='ij')
    norm = mcolors.LogNorm(vmin=max(spectral_values.min(), 1e-4), vmax=spectral_values.max() )
    #plt.figure(figsize=(7, 5))
    plt.scatter(Q_mesh.flatten(), Omega_mesh.flatten(), c=spectral_values.flatten(), cmap='viridis', norm=norm, s=40)
    plt.colorbar(label=r"$S(q,\Omega)$")
    plt.xlabel(r"Momentum $q$",fontsize =20)
    plt.ylabel(r"Frequency $\Omega$",fontsize =20)
    plt.title(f"Dynamical Structure Factor L={L}, N={N}, U={U}," + r' $\Gamma=$' + f'{Gamma}',fontsize = 20)
    plt.tight_layout()
    #plt.xlim(-0.5,0.5)
    #plt.ylim(-0.5,0.5)
    plt.show()

def dyn_structure_factor_1D(q_vals, Omega_vals, Bogo_spec, condensate, Bogo_eigenvectors,L, Gamma=0.25):
    L = len(condensate)
    u_values, v_values = symplectic_normalization_1D(Bogo_eigenvectors,L)
    print("goldstone mode !!", u_values[:,0])
    phi_star_j = np.conj(condensate)[:, None, None, None]
    phi_j = condensate[:, None, None, None]
    vj_mu = v_values[:, :, None, None]  # index j, mu, q, omega
    u_star_j_mu = np.conj(u_values)[:, :, None, None]
    phase_factor = np.exp(1j * q_vals[None, None, :, None] * np.arange(L)[:, None, None, None])
    denominator = (Omega_vals[None, None, :] - Bogo_spec[:, None, None] + 1j * Gamma)
    super_oscillator_strength = phase_factor * (phi_star_j * vj_mu + phi_j * u_star_j_mu)
    oscillator_strength = np.abs(np.sum(super_oscillator_strength, axis=0))**2
    spec_matrix = np.sum(oscillator_strength / denominator, axis=0)
    return -2 / L**2 * spec_matrix.imag

def dyn_structure_factor_1D_plot(f, q_vals, Omega_vals, Bogo_spec, condensate, Bogo_eigenvectors,L, Gamma=0.25):
    spectral_values = f(q_vals, Omega_vals, Bogo_spec, condensate, Bogo_eigenvectors,L, Gamma)
    Q_mesh, Omega_mesh = np.meshgrid(q_vals, Omega_vals, indexing='ij')
    norm = mcolors.LogNorm(vmin=max(spectral_values.min(), 1e-4), vmax=spectral_values.max())
    
    plt.scatter(Q_mesh.flatten(), Omega_mesh.flatten(), c=spectral_values.flatten(), cmap='viridis', norm=norm, s=40)
    exact_spectrum = bogo_spec(q_vals, U, n)
    plt.plot(q_vals, exact_spectrum, color='red', linewidth=2, label=r'Exact Spectrum')  

    #plt.colorbar(label=r"$S(q,\Omega)$")
    cbar = plt.colorbar(label=r"$S(q,\Omega)$")
    cbar.set_label(r"$S(q,\Omega)$", fontsize=15) 

    plt.xlabel(r"Momentum $q$", fontsize=15)
    plt.ylabel(r"Frequency $\Omega$", fontsize=15)
    plt.title(f"Dynamical Structure Factor L={L}, N={N}, U={U}," + r' $\Gamma=$' + f'{Gamma}', fontsize=15)
    plt.xticks(size=15)
    plt.yticks(size=15)
    #plt.xlim(-0.5,0.5)
    #plt.ylim(-0.5,0.5)
    plt.tight_layout()
    plt.show()

#dyn_structure_factor_1D_plot(dyn_structure_factor_1D, q_vals, Omega_vals, Bogo_spec, condensate, Bogo_eigenvectors,L, Gamma)


def WVrenorm(condensate, Bogo_eigenvectors,L):
    "m=1,2   j= 0,...,L -1    i = 0,..., 2L -1  "
    " eigenvectors_dyn is a collection of column (u v^*) it's a( L-1) x(L-1) matrix"
    u, v = symplectic_normalization_1D(Bogo_eigenvectors,L)  # normalize last L-1 columns only
    W = np.conj(condensate)[:,None]* u + condensate[:,None]* np.conj(v)
    V = np.conj(u[:,None,:])*u[:,:,None] + np.conj(v[:,:,None])*v[:,None,:]
    renorm = abs(condensate)**2 + np.sum(np.abs(v)**2, axis=1)
    return W,V,renorm

#WVrenorm(condensate,eigenvalues_dyn,eigenvectors_dyn)
print('WVRENORM')
W,V,renorm = WVrenorm(condensate, Bogo_eigenvectors,L)

def impur_kin(k):
    return -2*np.cos(k) +2
def Chevy_mat(condensate,Bogo_spec,W,V,renorm, g):
    'I organise (l,i)  as (0, 0) (0,1) (0,2) ... (0,L-1) (1,0) (1,1) ... (1,L-1).... (L-1,0) ... (L-1,L-1)'
    ' ok so (j,index = i) = (whatever,0) then no bogo excitations  and (whatever, 1-->L) is with bogo excitation'
    #matr = np.zeros(( L + L**2, L + L**2 ),dtype = complex)
    matr = np.zeros((  L**2,  L**2 ),dtype = complex)
    #print('W V renorm',W,V,renorm)
    for j in range(L):
        for j_prime in range(L):
            for index in range(L ):
                for index_prime in range(L ):
                    cste =  g*renorm[j] if (j==j_prime  and index ==index_prime) else 0 
                    #bath_kin = bogo_spec(2*np.pi/L*index,U,n)  if (j==j_prime and index ==index_prime  and index !=0) else 0
                    bath_kin = Bogo_spec[(index-1)%L ] if (j==j_prime and index ==index_prime  and index !=0) else 0

                    #impurity_kin = sum(impur_kin(2*np.pi/L*k)*np.exp(1j*(j-j_prime)*2*np.pi/L*k)/L for k in range(L) ) if (index==index_prime) else 0
                    impurity_kin = 0
                    if index == index_prime:
                        if (j + 1) % L == j_prime or (j - 1) % L == j_prime:
                            impurity_kin = -1
                        elif j == j_prime:
                            impurity_kin = 2 

                    W_int = 0
                    if index ==0 and index_prime !=0:
                        W_int =g* W[j,(index_prime-1)%L] if (j==j_prime) else 0
                    elif index_prime ==0 and index !=0:
                        W_int =g* np.conj(W[j,(index-1)%L]) if (j_prime == j) else 0
                    V_int =g *V[j,(index_prime-1)%L, (index-1)%L] if (j == j_prime and index !=0 and index_prime != 0) else 0
                    #bath_kin = 0
                    #impurity_kin =0
                    #V_int = 0
                    #W_int =0
                    #cste =0
                    matr[j*(L)+ index, j_prime*(L) +index_prime] = cste + bath_kin + impurity_kin + W_int + V_int
    return matr


chevy_matrix = Chevy_mat(condensate,Bogo_spec,W,V,renorm, g)
#print('chev mat',chevy_matrix)
print('Hermiticity', np.max(np.abs(chevy_matrix - np.conj(chevy_matrix).T)))
eigenvalues_chev, eigenvectors_chev = np.linalg.eigh(chevy_matrix) #spectrum of the chevy
print('LOWEST eigenvalues chevy 1D', eigenvalues_chev[0],'\n', eigenvectors_chev[:,0] )

#for i in range(len(eigenvalues_chev)):
#    print('CHEVY EIGENVECTOR', np.round(eigenvectors_chev[:,i],2).reshape(L, L) , '\n')
print('dimension !!', len(dyn_mat))
print('dressed impurity', np.round(eigenvectors_chev,2))

def spec_func_1D(Q,Omega, eigenvalues_chev, eigenvectors_chev):
    eps = 0.25
    summation = 0
    for i in range(len(eigenvalues_chev)):
        #psi_j = [ eigenvectors_chev[:,i][ k*(L+1) ] for k in range(L) ]
        #oscillator_strength = np.abs(eigenvectors[:,i][ m-1 ])**2  
        oscillator_strength = abs(sum(eigenvectors_chev[:,i][ j*(L) ]*1/np.sqrt(L)*np.exp(-1j*Q*j ) for j in range(L))) **2
        #print('oscillator strength',oscillator_strength)
        summation += oscillator_strength/(Omega - eigenvalues_chev[i]   + 1j*eps)
    return -2*summation.imag

def spec_func_1D_full(Q_vals, Omega_vals, eigenvalues_chev, eigenvectors_chev, L, Gamma):
    j_vals = np.arange(L)[None, None, :, None]
    phase = np.exp(-1j * Q_vals[:, None, None, None] * j_vals)
    phase /= np.sqrt(L)
    num_states = eigenvectors_chev.shape[1]
    psi_j_n = eigenvectors_chev[::L, :][None, None, :, :]
    print('shape of psi jn  and phase', psi_j_n.shape,phase.shape )
    total_amp = np.sum(phase.conj().T *psi_j_n, axis=2)
    oscillator_strengths = np.abs(total_amp) ** 2
    denom = Omega_vals[None, :, None] - eigenvalues_chev[None, None, :] + 1j * Gamma
    oscillator_strengths = np.tile(oscillator_strengths, (1, 100, 1, 1))  # Shape becomes (1, 100, 20, 400)
    spectral_values = oscillator_strengths / denom  # Should now work for broadcasting
    brightest_labels = np.argmax(spectral_values.imag, axis=2)
    brightest_eigenstates = eigenvectors_chev[:, brightest_labels]
    result = -2 * np.sum(spectral_values.imag, axis=2)
    
    return result, brightest_labels, brightest_eigenstates
def spec_func_1D_full(Q_vals, Omega_vals, eigenvalues_chev, eigenvectors_chev, L, Gamma):
    N_bogo = eigenvectors_chev.shape[1]
    j_vals = np.arange(L)
    phase_factors = np.exp(-1j * Q_vals[:, None] * j_vals[None, :]) / np.sqrt(L)

    # Assume eigenvectors_chev shape is (L * L, N_bogo)
    # Chevy basis: index = j * L + nu → so nu=0 means index = j * L
    psi_no_bogo = eigenvectors_chev[::L, :]  # shape (L, N_bogo)
    
    total_amplitudes = phase_factors @ psi_no_bogo  # shape (len(Q), N_bogo)
    oscillator_strengths = np.abs(total_amplitudes) ** 2

    denom = Omega_vals[None, :, None] - eigenvalues_chev[None, None, :] + 1j * Gamma
    spectral_values = oscillator_strengths[:, None, :] / denom
    A = -2 * np.sum(spectral_values.imag, axis=2)

    brightest_table = []
    for q_idx, Q in enumerate(Q_vals):
        brightest_i = np.argmax(oscillator_strengths[q_idx])
        state_vec = eigenvectors_chev[:, brightest_i]
        Z = oscillator_strengths[q_idx, brightest_i]
        energy = eigenvalues_chev[brightest_i]
        brightest_table.append([state_vec, Z, energy, Q])

    return A, brightest_table


Q_vals = 2*np.pi/L*np.arange(L) -np.pi 
Q= 0
#print('spec Q', spec_func_1D(0,Omega, eigenvalues_chev, eigenvectors_chev), spec_func_1D(np.pi,Omega, eigenvalues_chev, eigenvectors_chev))
Omega_vals = np.linspace(0,15,10*L)
Omega_vals = np.linspace(-10,10,10*L)
#y =  spec_func_1D(Q,Omega_vals,eigenvalues_chev, eigenvectors_chev)
#plt.plot(Q,y)
#plt.plot(Omega_vals,y)
#plt.show()

def spectral_function_plot_new(f, Q_vals,Omega_vals,U,n,g,L):
    Q_mesh, Omega_mesh = np.meshgrid(Q_vals, Omega_vals, indexing='ij')  
    times = time.time() 
    #spectral_values = np.array([ [f(Q, Omega, eigvals, eigvecs)  for Omega in Omega_vals ] for Q, (eigvals, eigvecs) in zip(Q_vals, [np.linalg.eigh(Frolich_Hamiltonian_new(Q, U, n, g, L, multiple)) for Q in Q_vals])])  
    spectral_values = np.array([ [f(Q, Omega, eigvals, eigvecs)  for Omega in Omega_vals ] for Q, (eigvals, eigvecs) in zip(Q_vals, [np.linalg.eigh(Chevy_mat(condensate,Bogo_spec,W,V,renorm, g)) for Q in Q_vals])])
    
    print('spectral function vs omega vs Q took', time.time() - times, ' seconds')
    norm = mcolors.LogNorm(vmin=max(spectral_values.min(), 1e-6), vmax=spectral_values.max())
    plt.scatter(Q_mesh, Omega_mesh, c=spectral_values, cmap='viridis', norm=norm, s=35)
    plt.colorbar(label=r"$\mathcal{A}(\Omega,Q)$")
    plt.xlabel("impurity momentum Q")
    plt.ylabel(r"$\Omega$")
    plt.title(f'g = {g},' + f' broadening =' + f'{eps}, L={L}, U= {U},n = {n}' )
    plt.show()


def spectral_function_plot_1D(Q_vals, Omega_vals, spectral_values, g, L, Gamma):
    Q_mesh, Omega_mesh = np.meshgrid(Q_vals, Omega_vals, indexing='ij')
    
    norm = mcolors.LogNorm(vmin=max(spectral_values.min(), 1e-6), vmax=spectral_values.max())
    
    plt.figure(figsize=(8, 6))
    plt.scatter(Q_mesh, Omega_mesh, c=spectral_values, cmap='viridis', norm=norm, s=40)
    plt.colorbar(label=r"$\mathcal{A}(\Omega,Q)$")
    plt.xlabel("Impurity momentum $Q$")
    plt.ylabel(r"$\Omega$")
    plt.title(f"1D Spectral Function: $g = {g}, \Gamma = {Gamma}, L = {L}$")
    plt.tight_layout()
    plt.show()

##spec_vals, brightest_eigenstates = spec_func_1D_full(Q_vals, Omega_vals, eigenvalues_chev, eigenvectors_chev, L, Gamma)
#spec_vals= spec_func_1D_full(Q_vals, Omega_vals, eigenvalues_chev, eigenvectors_chev, L, Gamma)

##spectral_function_plot_1D(Q_vals, Omega_vals, spec_vals, g, L, Gamma)

print('COMPUTE THE SPECTRAL FCT VS OMEGA & Q')
#spectral_function_plot(spec_func_1D, Q_vals,Omega_vals,eigenvalues_chev, eigenvectors_chev,g)
#spectral_function_plot_new(spec_func_1D, Q_vals,Omega_vals,U,n,g,L)



def compute_g2(i, chevy_vector, V, W, phi, L):
    g2 = 0
    psi = chevy_vector.reshape(L, L)  # psi[j, nu]
    
    for j in range(L):
        phi_sq = np.abs(phi[(j + i) % L])**2
        g2 += phi_sq * np.abs(psi[j, 0])**2
        for nu in range(L - 1):
            g2 += phi_sq * np.abs(psi[j, nu + 1])**2
            g2 += W[(j + i) % L, nu] * np.conj(psi[j, 0]) * psi[j, nu + 1]
            g2 += np.conj(W[(j + i) % L, nu]) * np.conj(psi[j, nu + 1]) * psi[j, 0]
            for mu in range(L - 1):
                g2 += V[(j + i) % L, nu, mu] * np.conj(psi[j, mu + 1]) * psi[j, nu + 1]
    g2 /= L
    return g2

def compare_g2_three_states_1D(Q_list, energy_windows,
                               eigenvalues_chev, eigenvectors_chev,
                               W, V, phi, L, U, g, N,
                               colors=["red", "purple", "green"]):
    
    plt.figure(figsize=(8, 5))
    j_vals = np.arange(L)
    
    for idx, (Q, (Omega_min, Omega_max)) in enumerate(zip(Q_list, energy_windows)):
        N_chev = eigenvalues_chev.shape[0]
        phase_factors = np.exp(-1j * Q * j_vals) / np.sqrt(L)

        psi_j0 = eigenvectors_chev[np.arange(L)*L + 0, :] 
        total_amplitudes = phase_factors @ psi_j0
        Z_n = np.abs(total_amplitudes) ** 2

        energy_mask = (eigenvalues_chev >= Omega_min) & (eigenvalues_chev <= Omega_max)
        if not np.any(energy_mask):
            print(f"No states in energy window [{Omega_min}, {Omega_max}] for Q = {Q}")
            continue

        brightest_index = np.where(energy_mask)[0][np.argmax(Z_n[energy_mask])]
        psi_max = eigenvectors_chev[:, brightest_index]

        g2 = np.zeros(L)
        for i in range(L):
            g2[i] = compute_g2(i, psi_max, V, W, phi, L)

        label = f"$Q$ = {Q:.2f}, $E$ = {eigenvalues_chev[brightest_index]:.2f}, $Z$ = {Z_n[brightest_index]:.2f}"
        #plt.plot(j_vals, g2.real, marker='o', color=colors[idx], label=label)
        #g2_real = g2.real
        #g2_norm = (g2_real - np.min(g2_real)) / (np.max(g2_real) - np.min(g2_real))
        #plt.plot(j_vals, g2_norm, marker='o', color=colors[idx], label=label + " (rescaled)")

        g2_normalized = g2.real / np.max(np.abs(g2.real))
        plt.plot(j_vals, g2_normalized, marker='o', color=colors[idx], label=label )


    plt.xlabel("Site $j$", fontsize=35)
    plt.ylabel(r"$g^{(2)}$", fontsize=35)
    plt.title(f"$U$ = {U}, $g$ = {g}, $L$ = {L}, $N$ = {N}", fontsize=35)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.legend(fontsize=15)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

Q_list = [0, 0, np.pi]  
#Q_list = [0, 0, 0]  
energy_windows = [(-9, -5), (-5, -2.2), (2, 5)] 
#energy_windows = [(0, 2), (2.5, 4), (12, 15)] 

#compare_g2_three_states_1D(   Q_list=Q_list,   energy_windows=energy_windows,  eigenvalues_chev=eigenvalues_chev,   eigenvectors_chev=eigenvectors_chev,   W=W, V=V, phi=renorm, L=L,    U=U, g=g, N=N)






def analyze_brightest_chevy_state_1D(Q, Omega_min, Omega_max,
                                     eigenvalues_chev, eigenvectors_chev,
                                     W, V, phi, L, Gamma=0.25, plot=True):

    N_chev = eigenvalues_chev.shape[0]
    j_vals = np.arange(L)
    phase_factors = np.exp(-1j * Q * j_vals) / np.sqrt(L)

    psi_j0 = eigenvectors_chev[np.arange(L)*L + 0, :] 
    total_amplitudes = phase_factors @ psi_j0
    Z_n = np.abs(total_amplitudes) ** 2

    energy_mask = (eigenvalues_chev >= Omega_min) & (eigenvalues_chev <= Omega_max)
    if not np.any(energy_mask):
        print(f"No states in energy window [{Omega_min}, {Omega_max}]")
        return None

    brightest_index = np.where(energy_mask)[0][np.argmax(Z_n[energy_mask])]
    psi_max = eigenvectors_chev[:, brightest_index]

    psi = psi_max.reshape(L, L)
    rho = np.sum(np.abs(psi)**2, axis=1)
    #rho /= np.sum(rho)

    g2 = np.zeros(L)
    for i in range(L):
        g2[i] = compute_g2(i, psi_max, V, W, phi, L)

    if plot:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(np.arange(L), rho, marker='o', color='red')
        #plt.title("Impurity Density")
        plt.xlabel("Site j",size=20)
        plt.ylim(0,0.06)
        plt.ylabel(r"$\rho_{imp}$",size =20)
        plt.xticks(fontsize =15)
        plt.yticks(fontsize =15)
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(np.arange(L), g2.real, marker='x', color='green')
        #plt.title(r"$g^{(2)}_i$")
        plt.xlabel("Site j",size = 20)
        plt.ylabel(r"$g^{(2)}$",size =20)
        plt.xticks(fontsize =15)
        plt.yticks(fontsize =15)
        plt.grid(True)

        plt.suptitle(
            f"Brightest Chevy state : Q={Q:.2f}, Ω∈[{Omega_min:.2f}, {Omega_max:.2f}], "
            f"E={eigenvalues_chev[brightest_index]:.4f}, Z={Z_n[brightest_index]:.4f}, U={U}, g={g}, L={L}, N={N}"
        ,size =20)
        plt.tight_layout()
        plt.show()

    return {
        "rho": rho,
        "g2": g2,
        "Z": Z_n[brightest_index],
        "energy": eigenvalues_chev[brightest_index],
        "state_index": brightest_index }

#result = analyze_brightest_chevy_state_1D(  Q=2, Omega_min=8, Omega_max=12, eigenvalues_chev=eigenvalues_chev, eigenvectors_chev=eigenvectors_chev,   W=W, V=V,   phi=renorm,L=L)  # or phi=phi if you named it that   L=L,   Gamma=0.25,   plot=True)





















def compute_g2(i, chevy_vector, V, W, phi, L):
    g2 = 0
    psi = chevy_vector.reshape(L, L)

    for j in range(L):
        phi_sq = np.abs(phi[(j + i) % L])**2
        g2 += phi_sq * np.abs(psi[j, 0])**2
        for nu in range(L - 1):
            g2 += phi_sq * np.abs(psi[j, nu + 1])**2
            g2 += W[(j + i) % L, nu] * np.conj(psi[j, 0]) * psi[j, nu + 1]
            g2 += np.conj(W[(j + i) % L, nu]) * np.conj(psi[j, nu + 1]) * psi[j, 0]
            for mu in range(L - 1):
                g2 += V[(j + i) % L, nu, mu] * np.conj(psi[j, mu + 1]) * psi[j, nu + 1]
    g2 /= L
    return g2


def impurity_density(chevy_vector, L):

    psi = chevy_vector.reshape(L, L)
    return np.sum(np.abs(psi)**2, axis=1)


def plot_g2_vs_i(L, chevy_vector, V, W, phi, spectral_value, energy, Q=None):
    g2_values = []

    for i in range(L):
        g2_value = compute_g2(i, chevy_vector, V, W, phi, L)
        g2_values.append(g2_value)

    plt.plot(range(L), g2_values, marker='o', linestyle='-', color='b')
    plt.xlabel('i', fontsize=14)
    plt.ylabel(r'$g^{(2)}_i$', fontsize=14)
    title = rf'$g^{(2)}_i$ for Z= {spectral_value:.3f}, E = {energy:.3f}'
    if Q is not None:
        title += rf', $Q = {Q:.2f}$'
    plt.title(title, fontsize=16)
    plt.grid(True)
    plt.show()


def plot_density_polaron(density_pol, spectral_value, osc_strength, energy, Q=None):
    plt.plot(range(len(density_pol)), density_pol, marker='o', linestyle='-', color='red')
    plt.xlabel('i', fontsize=30)
    plt.ylabel('Polaron density', fontsize=30)
    plt.ylim(-0.1, 0.6)
    title = rf'Polaron density  $Z = {osc_strength:.3f}$, $E = {energy:.3f}$'
    if Q is not None:
        title += rf', $Q = {Q:.2f}$'
    plt.title(title, fontsize=20)
    plt.grid(True)
    plt.show()


A, table = spec_func_1D_full(Q_vals, Omega_vals, eigenvalues_chev, eigenvectors_chev, L, Gamma)

state = table[L//2] 

chevy_vector = state[0]
Z_val = state[1]
energy = state[2]
Q_val = state[3]

#density_pol = impurity_density(chevy_vector, L)
##plot_density_polaron(density_pol, Z_val  , Z_val , energy, Q_val)
##plot_g2_vs_i(L, chevy_vector, V, W, renorm, Z_val, energy, Q=Q_val)











def compute_g2(i,chevy_vector, V, W, phi, L):
    g2 = 0
    psi = chevy_vector.reshape(L, L) 
    #psi = chevy_vector
    for j in range(L):
        g2 += np.abs(phi[(j+i)%L])**2*np.conj(psi[j, 0])*psi[j, 0]
        for nu in range(L-1):
            g2 += np.abs(phi[(j+i)%L])**2*np.conj(psi[j, nu+1])*psi[j, nu+1]
            g2 += W[(j+i)%L, nu] * np.conj(psi[j, 0])*psi[j, nu+1]
            g2 += np.conj(W[(j+i)%L, nu])*np.conj(psi[j, nu+1])*psi[j, 0]
            for mu in range(L-1):
                g2 += V[(j+i)%L, nu, mu]*np.conj(psi[j, mu+1])*psi[j, nu+1]
    g2 /= L
    return g2
def impurity_density(chevy_vector,L):
    psi = chevy_vector.reshape(L, L) 
    density = np.sum(np.abs(psi)**2, axis = 1)
    #density = [np.abs(psi[j,0])**2 for j in range(L)]
    return density


def plot_g2_vs_i(L, chevy_vector, V, W, phi):
    g2_values = []

    for i in range(L):
        g2_value = compute_g2(i, chevy_vector, V, W, phi, L)
        g2_values.append(g2_value)

    plt.plot(range(L), g2_values, marker='o', linestyle='-', color='b')
    plt.xlabel('i', fontsize=14)
    plt.ylabel(r'$g^{(2)}_i$', fontsize=14)
    plt.title(r'$g^{(2)}_i$ vs. i', fontsize=16)
    plt.grid(True)
    plt.show()

def plot_density_polaron(density_pol):
    
    plt.plot(range(L), density_pol, marker='o', linestyle='-', color='red')
    plt.xlabel('i', fontsize=30)
    plt.ylabel(r'Polaron density', fontsize=30)
    plt.ylim(-0.1,0.5)
    plt.grid(True)
    plt.show()


#chevy_vector= eigenvectors_chev[:,-1]
#print('max eigenstates',  np.round( brightest_eigenstates,2))
#chevy_vector=  brightest_eigenstates[:,0]
#print('chevy vector example!! when g=0', np.round(chevy_vector,2))
#density_pol = impurity_density(chevy_vector,L)
#print('density polaron', density_pol)
##plot_density_polaron(density_pol)
##plot_g2_vs_i(L, chevy_vector, V, W, renorm)
##############################################################################################################################################################################
########################################### Bogoliubov and Chevy in moemntum space for the ladder : lower band approximation #################################################
##############################################################################################################################################################################

#lb is for lower band , this is useful for Meissner phase and biased ladder

condensate = np.array([ 8.26733824e-01+1.31965758e-22j,  5.62593267e-01+1.32677894e-22j,
  7.86270591e-01-2.55474802e-01j,  5.35057992e-01-1.73850880e-01j,
  6.68841714e-01-4.85941950e-01j,  4.55147514e-01-3.30684025e-01j,
  4.85941950e-01-6.68841714e-01j,  3.30684025e-01-4.55147514e-01j,
  2.55474802e-01-7.86270591e-01j,  1.73850880e-01-5.35057992e-01j,
 -1.49580794e-22-8.26733824e-01j, -1.50387975e-22-5.62593267e-01j,
 -2.55474802e-01-7.86270591e-01j, -1.73850880e-01-5.35057992e-01j,
 -4.85941950e-01-6.68841714e-01j, -3.30684025e-01-4.55147514e-01j,
 -6.68841714e-01-4.85941950e-01j, -4.55147514e-01-3.30684025e-01j,
 -7.86270591e-01-2.55474802e-01j, -5.35057992e-01-1.73850880e-01j,
 -8.26733824e-01-3.09913581e-17j, -5.62593267e-01-7.16186341e-17j,
 -7.86270591e-01+2.55474802e-01j, -5.35057992e-01+1.73850880e-01j,
 -6.68841714e-01+4.85941950e-01j, -4.55147514e-01+3.30684025e-01j,
 -4.85941950e-01+6.68841714e-01j, -3.30684025e-01+4.55147514e-01j,
 -2.55474802e-01+7.86270591e-01j, -1.73850880e-01+5.35057992e-01j,
 -3.86978465e-21+8.26733824e-01j, -3.89066818e-21+5.62593267e-01j,
  2.55474802e-01+7.86270591e-01j,  1.73850880e-01+5.35057992e-01j,
  4.85941950e-01+6.68841714e-01j,  3.30684025e-01+4.55147514e-01j,
  6.68841714e-01+4.85941950e-01j,  4.55147514e-01+3.30684025e-01j,
  7.86270591e-01+2.55474802e-01j,  5.35057992e-01+1.73850880e-01j])
condensate = np.array([0.70710678 + 0.j, 0.70710678 + 0.j, 0.70710678 + 0.j, 0.70710678 + 0.j,
                        0.70710678 + 0.j, 0.70710678 + 0.j, 0.70710678 + 0.j, 0.70710678 + 0.j,
                        0.70710678 + 0.j, 0.70710678 + 0.j, 0.70710678 + 0.j, 0.70710678 + 0.j,
                        0.70710678 + 0.j, 0.70710678 + 0.j, 0.70710678 + 0.j, 0.70710678 + 0.j,
                        0.70710678 + 0.j, 0.70710678 + 0.j, 0.70710678 + 0.j, 0.70710678 + 0.j,
                        0.70710678 + 0.j, 0.70710678 + 0.j, 0.70710678 + 0.j, 0.70710678 + 0.j,
                        0.70710678 + 0.j, 0.70710678 + 0.j, 0.70710678 + 0.j, 0.70710678 + 0.j,
                        0.70710678 + 0.j, 0.70710678 + 0.j, 0.70710678 + 0.j, 0.70710678 + 0.j,
                        0.70710678 + 0.j, 0.70710678 + 0.j, 0.70710678 + 0.j, 0.70710678 + 0.j,
                        0.70710678 + 0.j, 0.70710678 + 0.j, 0.70710678 + 0.j, 0.70710678 + 0.j])


L = len(condensate)//2
n = sum( abs(condensate[i])**2 for i in range(len(condensate)) )/(2*L)

R = 3
U= 0
chi = np.pi/2
g = -5
multiple = 1
cutoff = multiple*2*np.pi

#dk =  0.1
dk = 2*np.pi/L
# we need a condensate and we need to compute the density by computing sum(|psi_{j,m}|^2 for j in range(L) for m in range(0,1))/(2*L)

k_values = [dk*i for i in range(int(round(cutoff/dk,0)))]
def chem_potential_lb_Meissner(U,n):
    return spec(R,1,0,chi) + U*n

def free_spec_lb_Meissner(k,R,chi,mu):
    return spec(R,1,k,chi) -mu


def spec_impurity_lb_Meissner(k,m,R,chi):  # chose chi =0 if uncharged
    return spec(R,m,k,chi) - spec(R,m,0,chi)


def reduced_spec_lb_Meissner(k,R,chi,U,n): # two body problem
    mu = chem_potential_lb_Meissner(U,n)
    return free_spec_lb_Meissner(k,R,chi,mu) + spec_impurity_lb_Meissner(k,R,chi) + U*n


def bogo_spec_lb_Meissner(k,R,chi,U,n):
    mu = chem_potential_lb_Meissner(U,n)
    Ek = free_spec_lb_Meissner(k,R,chi,mu)
    return np.sqrt(np.abs((Ek + 2*U*n)**2 -(U*n*np.sin(2*theta(k,R,chi)))**2))


def eta_lb_Meissner(k,R,chi,U,n):
    mu = chem_potential_lb_Meissner(U,n)
    Ek = free_spec_lb_Meissner(k,R,chi,mu)
    return 1/4*np.log(abs((Ek + U*n*(2+ np.sin(2*theta(k,R,chi))) )/(Ek + 3*U*n)))


def uk_lb_Meissner(k,m,R,chi,U,n): # m= 1,2 rail index
    m_term = np.sin(theta(k,R,chi) + np.pi/2*(1-m))
    eta = eta_lb_Meissner(k,R,chi,U,n)
    return np.cosh(eta) * m_term
def vk_lb_Meissner(k,m,R,chi,U,n):
    m_term = np.sin(theta(k,R,chi) + np.pi/2*(1-m))
    eta = eta_lb_Meissner(k,R,chi,U,n)
    return np.sinh(eta) * m_term


def Wk_lb_Meissner(k,m,R,chi,U,n):
    return uk_lb_Meissner(k,m,R,chi,U,n) + vk_lb_Meissner(-k,m,R,chi,U,n)
def Vkk_lb_Meissner(k,k_prime,m, R,chi,U,n):
    return  uk_lb_Meissner(k,m,R,chi,U,n)*  uk_lb_Meissner(k_prime,m,R,chi,U,n) +  vk_lb_Meissner(-k,m,R,chi,U,n)*  vk_lb_Meissner(-k_prime,m,R,chi,U,n)
#x = np.linspace(-np.pi,np.pi,1000)
#fx = bogo_spec_lb_Meissner(x,R,chi,U,n)
#fx = free_spec_lb_Meissner(x,R,chi,chem_potential_lb_Meissner(U,n))
#fx = theta(x,R,chi)
#fx = eta_lb_Meissner(x,R,chi,U,n)
#fx = uk_lb_Meissner(x,1,R,chi,U,n)
#fx = Wk_lb_Meissner(x,1,R,chi,U,n)
#fx =Vkk_lb_Meissner(x,2*np.pi/L,1,1, R,chi,U,n)
#plt.plot(x,fx)
#plt.show()

def Frolich_Hamiltonian_lb(condensate,p,chi,R,U,n,g,L,multiple = 1):
    """ condensate is a list of length 2L  |psi_{j,m}|^2=n_{j,m} = [ n_{0,1}, n_{0,2}, n_{1,1},n_{1,2}, n_{2,1}, n_{2,2}, n_{3,1}, n_{3,2},... ] in principle it should not depend on j for meissner and biased ladder
     I organise the tuple index =(k_1,m,m_1) = (0,1,0), (0,2,0) then (1,1,1),(1,1,2) , (1,2,1),  (1,2,2) then (2,1,1), (2,1,2), (2,2,1), (2,2,2)...etc   """
    H_Frolich = np.zeros((multiple*4*L -2,multiple*4*L -2),dtype = np.complex64)
    for index in range(multiple*4*L -2):
        for index_prime in range(multiple*4*L -2):
            k1 = (index +2)//4
            k1_prime = (index_prime +2)//4
            m, m1 = [(1, 1), (1, 2), (2, 1), (2, 2)][(index-2) % 4] if index >1 else [(1, False), (2, False)][index % 2] #m and m1 are rail indices for the impurity
            m_prime, m1_prime = [(1, 1), (1, 2), (2, 1), (2, 2)][(index_prime-2) % 4] if index_prime >1  else [(1, False), (2, False)][index_prime % 2]#m and m1 are rail indices for the impurity
            cste = 0
            if m == m_prime and k1 == k1_prime and k1 == 0:
                cste = g*( condensate[m-1] +1/L* sum(vk_lb_Meissner(2 * np.pi / L * k_id, m, R, chi, U, n) ** 2  for k_id in range(1, multiple * L)  ))
            elif m1 == m1_prime and k1 == k1_prime and k1 != 0:
                cste = g*(  condensate[m1-1] + 1/L* sum(vk_lb_Meissner(2 * np.pi / L * k_id, m1, R, chi, U, n) ** 2 for k_id in range(1, multiple * L)))
            impurity_kin =  0
            if k1 == k1_prime and k1 == 0:
                impurity_kin =   sum(1/2*(-1)**( m+ m_prime + s +(m+m_prime)*s +1)* spec_impurity_lb_Meissner(p,s,R,0) for s in range(1,3))
                #impurity = sum(np.cos( theta(p,R,chi) -np.pi/2*(m-s))* np.cos( theta(p,R,chi) -np.pi/2*(m_prime-s))*spec_impurity_lb_Meissner(p,s,R,chi) for s in range(1,3) ) 
            elif k1 == k1_prime and k1 != 0:
                impurity_kin =   sum(1/2*(-1)**( m1+ m1_prime + s +(m1+m1_prime)*s +1)* spec_impurity_lb_Meissner(p- 2*np.pi/L*k1,s,R,0) for s in range(1,3))
                #impurity = sum(np.cos( theta(p - 2*np.pi/L*k1,R,chi) -np.pi/2*(m1-s))* np.cos( theta(p - 2*np.pi/L*k1,R,chi) -np.pi/2*(m1_prime-s))*spec_impurity_lb_Meissner(p,s,R,chi) for s in range(1,3) ) 
            
            bath = bogo_spec_lb_Meissner(2*np.pi/L*k1,R,chi,U,n)  if ( k1==k1_prime and k!= 0 and m1 == m1_prime) else 0

            BI_int_W = 0
            if k1 == 0 and k1_prime != 0 and m == m1_prime:
                BI_int_W = g*np.sqrt(condensate[m-1]/L)*Wk_lb_Meissner(2*np.pi/L*k1_prime,m,R,chi,U,n)
            elif k1 != 0 and k1_prime == 0 and m_prime == m1:
                BI_int_W= g*np.sqrt(condensate[m1-1]/L)*Wk_lb_Meissner(2*np.pi/L*k1,m1,R,chi,U,n)
            V = Vkk_lb_Meissner(2*np.pi/L *k1,2*np.pi/L *k1_prime ,m1, R,chi,U,n) if (k1!= 0 and k1_prime != 0 and m1 == m1_prime) else 0
            BI_int_V =  g/L*V 
            H_Frolich[ index, index_prime] =  impurity_kin + bath + cste + BI_int_W + BI_int_V
    return H_Frolich

#herm_mat = Frolich_Hamiltonian_lb(condensate,0,chi,R,U,n,g,L)
#print('is it hermitian ?',np.max(np.abs(herm_mat -np.conj(herm_mat).T)))

def spectral_function_lb(Q,m,Omega,eigenvalues,eigenvectors):
    summation = 0
    for i in range(len(eigenvalues)):
        oscillator_strength = np.abs(eigenvectors[:,i][ m-1 ])**2  #m
        #oscillator_strength = np.abs(1/2*(eigenvectors[:,i][ 0 ]  +(-1)**(s)*eigenvectors[:,i][ 1])  )**2  # s
        summation += oscillator_strength/(Omega - eigenvalues[i]   + 1j*eps)
    return -2*summation.imag

Q=0
#eigenvalues,eigenvectors = np.linalg.eigh(herm_mat)
#Omega_vals =np.linspace(-10,10,400)
#y_vals = [spectral_function_lb(Q,m,Omega,eigenvalues,eigenvectors) for Omega in Omega_vals]
#plt.scatter(Omega_vals,y_vals)
#plt.xlabel(' Omega')
#plt.ylabel(' spectral function')
#plt.title(r'$\Omega = 0$'+ f' g={g}')
#print('This took', time.time()-time_start, 'seconds')
#plt.show()

#def spectral_function_plot_lb(f, Q_vals,Omega_vals,chi,R,U,n,g,L):
#    Q_mesh, Omega_mesh = np.meshgrid(Q_vals, Omega_vals, indexing='ij')  
#    start_time = time.time()
#    eig_data = [np.linalg.eigh(Frolich_Hamiltonian_lb(condensate, Q, chi, R, U, n, g, L, multiple)) for Q in Q_vals]

#    spectral_values1 = np.array([ [f(Q, 1, Omega, eigvals, eigvecs) for Omega in Omega_vals] for Q, (eigvals, eigvecs) in zip(Q_vals, eig_data)])
#    print(f'Spectral function m=1 vs omega vs Q took {time.time() - start_time:.2f} seconds')

#    spectral_values2 = np.array([[f(Q, 2, Omega, eigvals, eigvecs) for Omega in Omega_vals] for Q, (eigvals, eigvecs) in zip(Q_vals, eig_data)])
#    print(f'Spectral function m=2 vs omega vs Q took {time.time() - start_time:.2f} seconds')
#    fig, ax = plt.subplots(2, 1)
#    min_val = max(min(spectral_values1.min(), spectral_values2.min(), 1e-6), 1e-6)
#    max_val = max(spectral_values1.max(), spectral_values2.max())
#    norm = mcolors.LogNorm(vmin=min_val, vmax=max_val)
#    im1 = ax[0].scatter(Q_mesh, Omega_mesh, c=spectral_values1, cmap='viridis', norm=norm, s=35)
#    cb1 = fig.colorbar(im1, ax=ax[0])
#    cb1.set_label(r"$\mathcal{A}(\Omega,Q)$")
#    ax[0].set_xlabel("Impurity momentum Q")
#    ax[0].set_ylabel(r"$\Omega$")
#    ax[0].set_title(f'g = {g}, broadening = {m}, {eps}, L={L}, R={R}, U={U}, n={n}, $\chi$={round(chi,2)}')
#    im2 = ax[1].scatter(Q_mesh, Omega_mesh, c=spectral_values2, cmap='viridis', norm=norm, s=35)
#    cb2 = fig.colorbar(im2, ax=ax[1])
#    cb2.set_label(r"$\mathcal{A}(\Omega,Q)$")
#    ax[1].set_xlabel("Impurity momentum Q")
#    ax[1].set_ylabel(r"$\Omega$")
#    ax[1].set_title(f'g = {g}, broadening = {m}, {eps}, L={L}, R={R}, U={U}, n={n}, $\chi$={round(chi,2)}')
#    plt.show()
Q_vals = [2*np.pi/L*i for i in range(L) ]
Omega_vals = np.linspace(-10,10,20)

#spectral_function_plot_lb(spectral_function_lb, Q_vals,Omega_vals,chi,R,U,n,g,L)

def plot_spectral_function_plot_omega_g_new(f,inv_g_vals,Omega_vals,Q,chi,R,U,n,L): 
    inv_t_vals = inv_ak_vals
    inv_g_vals = inv_t_vals 

    #inv_g_Lippmann_Schwinger = [1/lippmann_Schwinger_g(k_values_3D,om,R,chi,U,n,L) for om in Omega_vals[:len(Omega_vals)//2]]
    #alt
    inv_g_Lippmann_Schwinger = [lippmann_Schwinger_g(k_values_3D,om,R,chi,U,n,L) for om in Omega_vals[:len(Omega_vals)//2]]
    
    #polariz_bubble = -1/L*sum(1/(reduced_spec(m_I,m_B,k_values_3D[id],R,chi,U,n)  )for id in range(1,len(k_values_3D)) )

    #inv_t_Lippmann_Schwinger = [ inv_g - polariz_bubble for inv_g in inv_g_Lippmann_Schwinger]
    #inv_t_Lippmann_Schwinger = inv_g_Lippmann_Schwinger
    #alt
    #inv_t_Lippmann_Schwinger = [ 1/(1/inv_g - polariz_bubble) for inv_g in inv_g_Lippmann_Schwinger]
    inv_t_Lippmann_Schwinger = inv_g_Lippmann_Schwinger
    inv_ak_Lippmann_Schwinger =[el  for el in inv_t_Lippmann_Schwinger ]

    spectral_values1 = np.zeros( (  (len(Omega_vals)), len(inv_ak_vals) ) )
    spectral_values2 = np.zeros( (  (len(Omega_vals)), len(inv_ak_vals) ) )

    Omega_mesh , inv_ak_mesh  = np.meshgrid( Omega_vals, inv_ak_vals,indexing='ij')  
    time_s= time.time()
 
    for i in range(len(inv_g_vals)):
        #Froch = Frolich_Hamiltonian_3D(Q, chi, R, U, n, 1/inv_g_vals[i], L)
        #Froch = Frolich_Hamiltonian_new(Q,chi,R,U,n,1/inv_g_vals[i],L,multiple)
        #alt
        Froch = Frolich_Hamiltonian_lb(condensate,Q,chi,R,U,n,g,L,multiple )
        
        eigenvalues, eigenvectors = np.linalg.eigh(Froch)
        spectral_values1[:, i] = np.array([f(Q,1,x,eigenvalues,eigenvectors )
                                for x in Omega_vals] )
        spectral_values2[:, i] = np.array([f(Q,2,x,eigenvalues,eigenvectors )
                                for x in Omega_vals] )
        print('1/(ka) =', inv_ak_vals[i], " it took", time.time()-time_s,' seconds')
    norm = mcolors.LogNorm(vmin=max(spectral_values1.min(),spectral_values2.min() ,1e-6), vmax=max(spectral_values1.max(),spectral_values2.max()))
    plt.plot(inv_ak_Lippmann_Schwinger,Omega_vals[:len(Omega_vals)//2],linestyle='-',color='red',lw=2)
    plt.xlim(min(inv_ak_vals),max(inv_ak_vals))
    plt.ylim(min(Omega_vals),max(Omega_vals))
    plt.scatter( inv_ak_mesh,Omega_mesh, c=spectral_values2, cmap='viridis', norm=norm, s=35)
    plt.colorbar(label=r"$\mathcal{A}(\Omega,Q)$")
    plt.title(f'Q={Q}, broadening =' + f'{eps}, L={L}, R ={R}, U= {U},n = {n}' + r',$\chi$=' + f'{round(chi,2)}')
    plt.ylabel(r"$\Omega$")
    #plt.xlabel(r'$1/g$')
    plt.xlabel(r'$g$')
    plt.show()


Q=0
inv_ak_vals = np.linspace(-1,1,50)
Omega_vals = np.linspace(-1,1,50)
timee = time.time()

#plot_spectral_function_plot_omega_g_new(spectral_function_lb,inv_ak_vals,Omega_vals,Q,chi,R,U,n,L)

def Bogo_spectrum_full(k,m,R,chi,U,n): # Do a version for meissner BL and vortex pls
    """ m = 1,2 """
    return None


def Frolich_Hamiltonian_full(chi,R,U,n,g,L):
    """ index orders no bogo + (k1,m1,m2) explicitely as no bogo +  (1,1,1) (1,1,2) (1,2,1) (1,2,2) (2,1,1)...  
    p1 orders (p,m) explicitely as (0,1) (0,2) (1,1) (1,2) (2,1) (2,2) (3,1) (3,2) (4,1).....
    """
    H_Frolich = np.zeros((2*L*(4*L-1),2*L*(4*L-1)),dtype = complex)
    cste2 = sum(sum(  vk(2*np.pi/L*(k_index),m_index,R,chi,U,n)**2 for k_index in range(1,L)) for m_index in range(1,3))
    for p1 in range(2*L):
        for index in range(4*L-1):
            for p1_prime in range(2*L):
                for index_prime in range(4*L-1):    
                    m = p1 % 2 +1                #rail index for impurity
                    m_prime = p1_prime % 2 + 1   
                    p = p1//2
                    p_prime = p1_prime//2
                    m1, m2 = [(1, 1), (1, 2), (2, 1), (2, 2)][(index-3) % 4] if index >2 else [(0, 0), (2, 1), (2, 2)][index % 3] # band index for Boglon and rail index for impurity
                    m1_prime,m2_prime =  [(1, 1), (1, 2), (2, 1), (2, 2)][(index_prime-3) % 4] if index_prime >2 else [(0, 0), (2, 1), (2, 2)][index_prime % 3] 
                    k = (index +1)//4  if index > 2 else 0
                    k_prime = (index_prime +1)//4  if index_prime > 2 else 0

                    impurity_kin = -2*np.cos(2*np.pi/L*(p-k)) + 2 if (p==p_prime and index == index_prime and index != 0 ) or (index ==0 and  index == index_prime and p1==p1_prime ) else 0
                    bath = Bogo_spectrum_full(2*np.pi/L*k,m1,R,chi,U,n) if (p==p_prime and index==index_prime and index!=0 ) else 0
                    BI_int_cste1 = g*n if (p==p_prime and index==index_prime and index !=0) or (p1==p1_prime and index==index_prime and index==0) else 0
                    BI_int_cste2 = g/L*cste2 if (p==p_prime and index==index_prime and index !=0) or (p1==p1_prime and index==index_prime and index==0) else 0
                    BI_int_W =0
                    if index ==0 and index_prime !=0:
                        BI_int_W = g*np.sqrt(n/L)*Wkm(2*np.pi/l*k_prime,m1_prime,R,chi,U,n) if (p==p_prime and m == m2_prime) else 0
                    elif index_prime ==0 and index !=0:
                        BI_int_W = g*np.sqrt(n/L)*Wkm(2*np.pi/l*k,m1,R,chi,U,n) if (p==p_prime and m_prime == m2)  else 0
                    BI_int_V = Vkmkm(k_prime,m1_prime,k,m1,R,chi,U,n) if (p==p_prime and index !=0 and index_prime !=0) else 0

                    H_Frolich[p1*(4*L-1) + index, p1_prime*(4*L-1) + index_prime] = impurity_kin + bath + BI_int_cste1 + BI_int_cste2 + BI_int_V + BI_int_W
    return H_Frolich

def Frolich_Hamiltonian_full_Q(Q,chi,R,U,n,g,L): #for one momentum sector
    """ index orders no bogo + (k1,m1,m2) explicitely as no bogo +  (1,1,1) (1,1,2) (1,2,1) (1,2,2) (2,1,1)...  
    p1 orders (p,m) explicitely as (0,1) (0,2) (1,1) (1,2) (2,1) (2,2) (3,1) (3,2) (4,1).....
    """
    H_Frolich = np.zeros((2*(4*L-1),2*(4*L-1)),dtype = complex)
    cste2 = sum(sum(  vk(2*np.pi/L*(k_index),m_index,R,chi,U,n)**2 for k_index in range(1,L)) for m_index in range(1,3))
    for p1 in range(2):
        for index in range(4*L-1):
            for p1_prime in range(2):
                for index_prime in range(4*L-1):    
                    m = p1 + 1
                    m_prime = p1_prime +1
                    m1, m2 = [(1, 1), (1, 2), (2, 1), (2, 2)][(index-3) % 4] if index >2 else [(0, 0), (2, 1), (2, 2)][index % 3] # band index for Boglon and rail index for impurity
                    m1_prime,m2_prime =  [(1, 1), (1, 2), (2, 1), (2, 2)][(index_prime-3) % 4] if index_prime >2 else [(0, 0), (2, 1), (2, 2)][index_prime % 3] 
                    k = (index +1)//4  if index > 2 else 0
                    k_prime = (index_prime +1)//4  if index_prime > 2 else 0

                    impurity_kin = -2*np.cos(2*np.pi/L*(p-k)) + 2 if (index == index_prime and index != 0 ) or (index ==0 and  index == index_prime and m==m_prime ) else 0
                    bath = Bogo_spectrum_full(2*np.pi/L*k,m1,R,chi,U,n) if (p==p_prime and index==index_prime and index!=0 ) else 0
                    BI_int_cste1 = g*n if (index==index_prime and index !=0) or (m==m_prime and index==index_prime and index==0) else 0
                    BI_int_cste2 = g/L*cste2 if (index==index_prime and index !=0) or (m==m_prime and index==index_prime and index==0) else 0
                    BI_int_W =0
                    if index ==0 and index_prime !=0:
                        BI_int_W = g*np.sqrt(n/L)*Wkm(2*np.pi/l*k_prime,m1_prime,R,chi,U,n) if (m == m2_prime) else 0
                    elif index_prime ==0 and index !=0:
                        BI_int_W = g*np.sqrt(n/L)*Wkm(2*np.pi/l*k,m1,R,chi,U,n) if (m_prime == m2) else 0
                    BI_int_V = Vkmkm(k_prime,m1_prime,k,m1,R,chi,U,n) if (index !=0 and index_prime !=0 and m2 == m2_prime) else 0

                    H_Frolich[p1*(4*L-1) + index, p1_prime*(4*L-1) + index_prime] = impurity_kin + bath + BI_int_cste1 + BI_int_cste2 + BI_int_V + BI_int_W
    return H_Frolich

#########################################################################################################################################################################################
########################################################################## Chevy in real space ladder ###################################################################################
#########################################################################################################################################################################################


#condensate = np.array([0.70710678 + 0.j, 0.70710678 + 0.j,0.70710678 + 0.j, 0.70710678 + 0.j])
J_parallel = 1
J_perp = 0.25
#J_perp = 0.2
#J_perp = 1.05
#J_perp = 0.1
#J_perp = 3
R = J_perp/J_parallel
U = 1
U=0.2
g=-5
p = 1/2
chi = np.pi/2*2*0.66
#chi = np.pi/2*1.99
chi = np.pi/2
L=4
N=L
n = N/(2*L)
Gamma = 0.1
b,c,condensate = GPE_condensate(R,chi,U,p,L,N,J_parallel, 0, 0,0, 0.01,800)

print('g,U,R,chi =', g,U,R,chi)
print("N=", N)
print('norm =', np.conj(condensate)@condensate)
print('b, omega =', b,c)
#print('FBZ', np.pi/(2*np.pi*b/L))

pow =   L//math.gcd(2*b, L) if c == np.pi/4 else 1 
print('transaltion pow', pow)
#print('L =',L)
#N = np.conj(condensate)@condensate

A1_R = A_m_Real(J_parallel, J_perp,U, chi,condensate,chi*(1-p),0)

A2_R = A_m_Real( J_parallel, J_perp,U,chi, condensate,-chi*p,1)
B1_R = B_m_Real( U, condensate,0)
B2_R = B_m_Real(U, condensate,1)
C1_R = C_m_Real(condensate, J_perp)
C2_R = C1_R
A1_R_hole = A_m_Real_hole(J_parallel,J_perp, U,chi, condensate,chi*(1-p),0)
A2_R_hole = A_m_Real_hole(J_parallel,J_perp, U, chi,condensate,-chi*p,1)


eom_matrix = construct_L(A1_R, B1_R, A2_R, B2_R, C1_R, C2_R,A1_R_hole,A2_R_hole,L,pow,0.000001)

eom_eigenvalues_unsorted, eom_eigenvectors_unsorted = np.linalg.eig(eom_matrix )
sorted_indices = np.argsort(eom_eigenvalues_unsorted)


Bogo_spec, uv_eigenvectors  = eom_eigenvalues_unsorted[sorted_indices][1+2*L:].real, eom_eigenvectors_unsorted[:, sorted_indices][:, 1+2*L:]
#Bogo_spec = np.array([0 for i in range(2*L-1)])
def Bogo_vs_k(eigenvalues,eigenvectors, L, pow, tol=1e-3):
    
    k_indices_raw = []
    band_energies_raw = []
    trans = T_matrix_canonical_basis(pow,L)
    for i in range(eigenvectors.shape[1]):
        vec = eigenvectors[:, i]
        k_raw = np.angle((np.conj(vec) @ trans)[0] * vec[0]) / pow
        if abs(k_raw-np.pi)< 0.0001:
            k_raw = -np.pi
        k_index = int(round(L * k_raw / (2 * np.pi)))
        k_indices_raw.append(k_index)
        band_energies_raw.append(eigenvalues[i].real)
    print('raw k raw bogo', k_indices_raw  ,'\n',band_energies_raw)
    unique_k = sorted(set(k_indices_raw))
    k_labels = list(unique_k)
    omega_bogo_list = [[] for _ in k_labels]
    for idx, k_i in enumerate(k_indices_raw):
        print('idx and ki', idx, k_i)
        k_pos = k_labels.index(k_i)  
        omega_bogo_list[k_pos].append(band_energies_raw[idx])
    for band_list in omega_bogo_list:
        band_list.sort()
    omega_bogo = np.array(omega_bogo_list) 
    return k_labels, omega_bogo

#k_labels, omega_bogo = Bogo_vs_k(eom_eigenvalues_unsorted[sorted_indices][2*L:].real,eom_eigenvectors_unsorted[:, sorted_indices][:, 2*L:], L, pow, tol=1e-3)
#print('klabels and omega bogo', k_labels,'\n', omega_bogo )
#print('raw bogo', eom_eigenvalues_unsorted[sorted_indices][2*L:].real)

print('bogo spec', Bogo_spec)

def WjmiVjmilrenormjm(condensate,uv_eigenvectors):
    "m=1,2   j= 0,...,L -1    i = 0,..., 2L -1   but the basis I chose for (j,m) was (0,0) (1,0) (2,0) .... (L-1,0) (0,1) (1,1) ..."
    " eigenvectors_dyn is a collection of column (u v^*) it's a 4L x 4L matrix"
    coh_coef = uv_eigenvectors
    #print('coh coef', coh_coef)
    u_unnormalized = coh_coef[:2*L,:]
    v_unnormalized = np.conj(coh_coef[2*L:,:])
    u_unnormalized = u_unnormalized.reshape(2,L, 2*L-1)        
    v_unnormalized = v_unnormalized.reshape(2,L,2*L-1)       
    phi =  condensate.reshape(L, 2).T   
    symplectic_norm = np.sqrt((np.abs(u_unnormalized)**2 - np.abs(v_unnormalized)**2)*2*L)
    u,v = u_unnormalized/ symplectic_norm, v_unnormalized/symplectic_norm
    W = np.conj(phi[:,:,None])*u[:,:,:] + phi[:,:,None]*np.conj(v[:,:,:])
    V = np.conj(u[:,:,:,None])*u[:,:,None,:] + np.conj(v[:,:,None,:] )*v[:,:,:,None]
    renorm = np.abs(phi)**2 + np.sum(np.abs(v)**2,axis = 2)
    #return None
    return W.transpose(1,0,2),V.transpose(1,0,2,3),renorm.T


def WjmiVjmilrenormjm(condensate, uv_eigenvectors):
    "m=1,2   j= 0,...,L -1    i = 0,..., 2L -1   but the basis I chose for (j,m) was (0,0) (1,0) (2,0) .... (L-1,0) (0,1) (1,1) ..."
    " eigenvectors_dyn is a collection of column (u v^*) it's a 4L x 4L matrix"
    L = len(condensate) // 2 
    u, v = symplectic_normalization_ladder(uv_eigenvectors)  
    u = u.reshape(2, L,  2*L-1).transpose(1, 0, 2)  #  u[j, m, i]
    v = v.reshape(2, L,  2*L-1).transpose(1, 0, 2)
    phi = condensate.reshape(L, 2) 
    W = np.conj(phi[:,:,None])*u[:,:,:] + phi[:,:,None]*np.conj(v[:,:,:]) 
    V = np.conj(u[:,:,:,None])*u[:,:,None,:] + np.conj(v[:,:,None,:])*v[:,:,:,None]
    renorm = np.abs(phi)**2 + np.sum(np.abs(v)**2, axis=2)
    return W, V, renorm

#WjmiVjmilrenormjm(condensate,eom_eigenvalues, eom_eigenvectors )
         
def impurity_spec_ladder(k,s,R,chi,L):
    "s= + is s= 1 and s=- is s= 2"
    return spec(R,s+1,k,chi) - spec(R,s+1,k_minus(R,chi,L),chi) 



def impurity_Chevy(Q_vals, R, chi, L):
    """
    returns a matrix ( m1m2, delta_j)=(4,2(2L) +1)
    """
    delta_j_vals =  np.arange(-(2 * L - 1), 2 * L)
    s_vals = np.arange(1,3)
    m = s_vals[:, None, None, None, None]   # (2,1,1,1,1)
    m_prime = s_vals[None, :, None, None, None]   # (1,2,1,1,1)
    delta_j = delta_j_vals[None, None, :, None, None]  # (1,1,D,1,1)
    s = s_vals[None, None, None, :, None]   # (1,1,1,2,1) 
    Q = np.array(Q_vals)[None, None, None, None, :]   #  (1,1,1,1,L)
    theta_vals = theta(Q, R, chi)  #  (1,1,1,1,L)
    total_mat = (1 / L  * np.exp(1j * delta_j * Q)  * np.cos(theta_vals - np.pi/2 * (m - s))  * np.cos(theta_vals - np.pi/2 * (m_prime - s)) *impurity_spec_ladder( Q,s,R,chi,L))
    impurity_matrix = total_mat.sum(axis=(3,4))  
    return impurity_matrix

def Frolich_real_space_old(condensate,eigenvalues_dyn,eigenvectors_dyn,W,V,renorm_density,L):
    " (pos,index) if index ==0, it's the no bogo part  "
    H_Frolich = np.zeros((2*L*(2*L),2*L*(2*L)),dtype = complex)
    for pos in range(2*L):
        j,i = pos//2, pos%2 
        for pos_prime in range(2*L):
            j_prime,i_prime = pos_prime//2, pos_prime%2 
            same_pos = pos == pos_prime
            for index in range(2*L):
                for index_prime in range(2*L):
                    same_index = index == index_prime
                    cste = g*renorm_density[j,i] if same_pos and same_index else 0
                    bath = eigenvalues_dyn[index -1] if (same_index and index !=0 and same_pos) else 0
                    impurity_kin = sum(sum(np.exp(1j*(j-j_prime)*2*np.pi/L*k)/L*np.cos(theta(2*np.pi/L*k,R,0) -np.pi/2*(i+1-s))*np.cos(theta(2*np.pi/L*k,R,0) -np.pi/2*(i_prime+1-s))*impurity_spec_ladder(2*np.pi/L*k,s,R,0,L) for s in range(1,3)) for k in range(L)) if same_index else 0
                    V_int = g*V[j,i,index -1 ,index_prime -1] if ( same_pos and index != 0 and index_prime !=0) else 0
                    W_int = 0
                    if index ==0 and index_prime !=0:
                        W_int = g*W[j,i,index_prime -1] if (same_pos) else 0
                    if index_prime ==0 and index !=0:
                        W_int = g*np.conj(W[j,i,index -1]) if (same_pos) else 0
                    #bath = 0
                    #W_int = 0
                    #V_int = 0
                    #cste = 0
                    H_Frolich[pos*(2*L) +index, pos_prime*(2*L) +index_prime] = cste + bath + impurity_kin + V_int + W_int
    return H_Frolich


def Frolich_real_space( Bogo_spec, W, V, renorm_density,R, L):
    size = 2*L*2*L  
    H_Frolich = np.zeros((size, size), dtype=complex)
    for pos in range(2 * L):
        j, i = pos // 2, pos % 2 
        for pos_prime in range(2 * L):
            same_pos = (pos == pos_prime)
            j_prime, i_prime = pos_prime//2,pos_prime%2  
            for index in range(2*L):
                for index_prime in range(2*L):
                    same_idx = (index == index_prime)
                    cste = g*renorm_density[j, i] if same_pos and same_idx else 0
                    bath = Bogo_spec[index - 1] if (same_pos and same_idx and index != 0) else 0
                    impurity_kin = 0
                    if same_idx :
                        if same_pos:
                            impurity_kin += (2 + R)  

                        if i == i_prime:
                            if (j + 1) % L == j_prime or (j - 1) % L == j_prime:  
                                impurity_kin = -1
                        elif j == j_prime and i != i_prime:  
                            impurity_kin = -R

                    V_int = g * V[j, i, index - 1, index_prime - 1] if (same_pos and index != 0 and index_prime != 0) else 0
                    W_int = 0
                    if same_pos:
                        if index == 0 and index_prime != 0:
                            W_int = g * W[j, i, index_prime - 1]
                        elif index_prime == 0 and index != 0:
                            W_int = g * np.conj(W[j, i, index - 1])
                    #bath =0
                    #V_int, W_int,bath= 0,0,0
                    V_int =0
                    W_int = 0
                    #if index !=0:
                    #    impurity_kin = 0
                    #impurity_kin =0
                    H_Frolich[pos*(2*L) + index, pos_prime*(2*L) + index_prime] = cste + bath + impurity_kin + V_int + W_int

    return H_Frolich

#Q_vals = [2*np.pi/L*i for i in range(L)] if c != np.pi/4 else [2*np.pi/L*i for i in range(L//(2*b))] 
Q_vals = [2*np.pi/L*i - np.pi for i in range(L)]
Omega_vals = np.linspace(-15,15,300)
Omega_vals = np.linspace(-4,15,300)
Omega_vals = np.linspace(-50,50,300)
#Omega_vals = np.linspace(-20,20,300)
#Omega_vals = np.linspace(-25,5,300)
#Omega_vals = np.linspace(0,30,1000)
time_s = time.time()
#impurity_matrix =impurity_Chevy(Q_vals, R, chi, L)
W,V,renorm_density =  WjmiVjmilrenormjm(condensate,uv_eigenvectors)
#print('W V renorm',W,V,renorm_density)
print(' time to compute W V renorm density ', time.time() - time_s)
#Chevy_ladder = Frolich_real_space(condensate,Bogo_spec,W,V,renorm_density,impurity_matrix,L)
Chevy_ladder = Frolich_real_space(Bogo_spec,W,V,renorm_density,R,L)
print('time to build the chevy mat ', time.time() - time_s, Chevy_ladder.shape)
#print('hermitian???', np.max(np.abs(Chevy_ladder - np.conj(Chevy_ladder).T)) )
#print('diff', np.max(np.abs(Chevy_ladder - np.conj(Chevy_ladder).T)))
eigenvalues_chev, eigenvectors_chev = np.linalg.eigh(Chevy_ladder)
print('time for chevy diagonalisation', time.time() - time_s)
for i in range(len(eigenvalues_chev)):
    print('eigenvalue index', i, '\n', eigenvalues_chev[i],'\n', eigenvectors_chev[:,i]  )

print('eigenvalue index 0   is ',   eigenvalues_chev[0],'\n', eigenvectors_chev[:,0]  )
def spectral_function_rho_plot(Q_vals, omega_vals, R, chi, L, Gamma, omega_bogo, k_labels, impurity_spec_func, mean_field_shift=0.0):
    num_Q = len(Q_vals)
    num_omega = len(omega_vals)
    num_bands_bogo = omega_bogo.shape[1]
    rho = np.zeros((num_omega, num_Q))  # shape: (omega, Q)
    
    k_vals = [2*np.pi*k_idx/L for k_idx in k_labels]
    #print('omega bogo', omega_bogo)
    for iQ, Q in enumerate(Q_vals):
        for ik, k in enumerate(k_vals):
            Q_minus_k = Q - k 
            for s in [1, 2]: 
                eps_imp = impurity_spec_func(Q_minus_k, s, R, chi, L)
                for sb in range(num_bands_bogo):
                    eps_bogo = omega_bogo[ik][sb]
                    #print('eps bogo', eps_bogo )
                    omega_sum = eps_imp + eps_bogo +mean_field_shift*g
                    lorentzian = 1/np.pi*Gamma/((omega_vals - omega_sum)**2 + Gamma**2)
                    rho[:, iQ] += lorentzian

    Omega_mesh, Q_mesh = np.meshgrid(omega_vals, Q_vals, indexing='ij')
    print('vmin and vmax values', max(rho.min(), 1e-6), rho.max())
    norm = mcolors.LogNorm(vmin=max(rho.min(), 1e-6), vmax=rho.max())

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(Q_mesh, Omega_mesh, c=rho, cmap='viridis', norm=norm, s=40)
    cbar = plt.colorbar(sc, label=r'$\rho(Q, \omega)$')
    cbar.set_label(label=r'$\rho(Q, \omega)$', fontsize=15) 
    plt.xlabel(r'$\mathrm{Q}$', size=15)
    plt.ylabel(r'$\omega$', size=15)
    plt.title(r'Spectral Function $\rho(Q,\omega)$', size=15)
    plt.tight_layout()
    plt.show()

    return rho



#print('Gamma and renorm ',Gamma, renorm_density[0,0])
#rho = spectral_function_rho_plot(Q_vals, Omega_vals, R, chi, L, Gamma, omega_bogo, k_labels, impurity_spec_ladder,renorm_density[0,0])





def spec_func_ladder(Q,s,Omega_vals, eigenvalues_chev, eigenvectors_chev,L,Gamma ):
    """ Outputs a list of spectral values for a given Q and a range of Omegas"""
    i_vals = np.arange(len(eigenvalues_chev))      # 1 x 4L^2       
    j_vals = np.arange(L)                              # 1 x L
    #print('L=',L)
    #print('j_vals',j_vals.shape)
    #cos_theta_Q = np.cos(theta(Q, R, 0)-np.pi/2*(1-s))   # 1 x 1  
    cos_theta_Q =1
    #sin_theta_Q = np.sin(theta(Q, R, 0)-np.pi/2*(1-s))   # 1 x 1
    sin_theta_Q = 0 # m=1
    phase_factors = np.exp(-1j * Q * j_vals) / np.sqrt(L)        #  1 x L
    psi_j1 = eigenvectors_chev[np.isin(i_vals,(2*np.arange(L))*2*L), :]  #L x 4L^2
    psi_j2 = eigenvectors_chev[np.isin(i_vals,(2*np.arange(L) +1)*2*L), :]   #L x 4L^2 
    #print('dim phase fac' ,phase_factors.shape)
    #print('dim psi j1' ,psi_j1.shape)
    #print('dim cos_theta_Q ' ,cos_theta_Q .shape)
    total_amplitudes = phase_factors@(psi_j1 * cos_theta_Q + psi_j2 * sin_theta_Q) #   1xL . L x 4L^2 
    oscillator_strengths = np.abs(total_amplitudes) ** 2       #  = Z_n for fixed Q dim    1x4L^2         
    spectral_values = oscillator_strengths / (Omega_vals[:, None] - eigenvalues_chev + 1j * Gamma)  #number of rows = len Omega_vals and number of columns : 4L^2
    summation = np.sum(spectral_values, axis=1)
    result = -2 *summation.imag
    return result 


def spectral_function_plot_ladder(f, Q_vals,s,Omega_vals,eigenvalues,eigenvectors,g,R,chi,U,L,Gamma ):
    timee = time.time()
    spectral_values = np.zeros( (  (len(Omega_vals)), len(Q_vals) ) )
    #print(' spec val intialisation ', time.time() - timee, ' seconds')
    #print('shape spec val', spectral_values.shape )
    Omega_mesh , Q_mesh= np.meshgrid( Omega_vals,Q_vals, indexing='ij')   
    #print(' Omega Q mesh', time.time() - timee, ' seconds')
    #spectral_values = f(Q_vals, s,Omega_vals, eigenvalues, eigenvectors,L ) 
    for i in range(len(Q_vals)):
        spectral_values[:,i] = f(Q_vals[i], s,Omega_vals, eigenvalues, eigenvectors,L,Gamma ) 
        #print('spec vals', f(Q_vals[i], s,Omega_vals, eigenvalues, eigenvectors,L,Gamma ) )
        print(f'Q ={Q_vals[i]} took ', time.time() - timee, ' seconds')
    #print('vmin and vmax,' ,max(spectral_values.min(), 1e-6), spectral_values.max())
    norm = mcolors.LogNorm(vmin=max(spectral_values.min(), 1e-6), vmax=spectral_values.max())
    #norm = mcolors.Normalize(vmin=spectral_values.min(), vmax=spectral_values.max())

    plt.scatter(Q_mesh, Omega_mesh, c=spectral_values, cmap='viridis', norm=norm, s=40)
    #plt.colorbar(label=r"$\mathcal{A}(\Omega,Q)$")
    #plt.colorbar(label=r"$S(q,\Omega)$")
    cbar = plt.colorbar(label=r"$\mathcal{A}(\Omega,Q)$")
    cbar.set_label(label=r"$\mathcal{A}(\Omega,Q)$", fontsize=30) 
    plt.xlabel("impurity momentum Q",size=30)
    plt.ylabel(r"$\Omega$",size=30)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.title(f'g = {g},' + r'$\Gamma =$'+ f'{Gamma}, L={L}, R ={R}, U= {U}, N= {round(N.real,1)}, m=1' + r',$\chi$=' + f'{round(chi,2)}',size=15)
    plt.show()

def spectral_function_plot_ladder_interpol(f, Q_vals, s, Omega_vals, eigenvalues, eigenvectors, g, R, chi, U, L, Gamma):
    Q_vals = np.array(Q_vals)
    Omega_vals = np.array(Omega_vals)
    timee = time.time()
    spectral_values = np.zeros((len(Omega_vals), len(Q_vals)))
    
    for i in range(len(Q_vals)):
        spectral_values[:, i] = f(Q_vals[i], s, Omega_vals, eigenvalues, eigenvectors, L, Gamma)
        print(f'Q = {Q_vals[i]} took', time.time() - timee, 'seconds')
    
    norm = mcolors.LogNorm(vmin=max(spectral_values.min(), 1e-6), vmax=spectral_values.max())

    # Create meshgrid for interpolation points
    Q_mesh, Omega_mesh = np.meshgrid(Q_vals, Omega_vals)

    # Flatten the data for griddata input
    points = np.array([Q_mesh.flatten(), Omega_mesh.flatten()]).T
    values = spectral_values.flatten()

    # Define a finer grid for interpolation
    Q_fine = np.linspace(Q_vals.min(), Q_vals.max(), 200)
    Omega_fine = np.linspace(Omega_vals.min(), Omega_vals.max(), 200)
    Q_fine_mesh, Omega_fine_mesh = np.meshgrid(Q_fine, Omega_fine)

    # Interpolate on the fine grid
    spectral_fine = griddata(points, values, (Q_fine_mesh, Omega_fine_mesh), method='cubic')

    plt.imshow(spectral_fine, extent=[Q_vals.min(), Q_vals.max(), Omega_vals.min(), Omega_vals.max()],
               origin='lower', aspect='auto', cmap='viridis', norm=norm)
    cbar = plt.colorbar(label=r"$\mathcal{A}(\Omega,Q)$")
    cbar.set_label(label=r"$\mathcal{A}(\Omega,Q)$", fontsize=30)
    plt.xlabel("impurity momentum Q", size=30)
    plt.ylabel(r"$\Omega$", size=30)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.title(f'g = {g}, ' + r'$\Gamma =$' + f'{Gamma}, L={L}, R ={R}, U= {U}, ' +
              r'$ \chi =$' + f'{round(chi,2)}', size=20)
    plt.show()



def spectral_function_vs_R(Q_fixed, R_vals, Omega_vals, s, g, L, Gamma):
    spectral_data = []
    eigenvalues_chev_dict = {}
    eigenvectors_chev_dict = {}
    U = 0.2
    J_parallel =1
    g=-5


    chi = np.pi/2 *1.5      

    N=L
    n = N/(2*L)
    Gamma = 0.1
    for R_val in R_vals:
        J_perp = R_val
        #print('R is', R_val)
        
        b,c,condensate = GPE_condensate(R,chi,U,p,L,N,J_parallel, 0, 0,0, 0.01,800)

        print('g,U,R,chi =', g,U,R_val,chi)
        print("N=", N)
        print('norm =', np.conj(condensate)@condensate)
        print('b, omega =', b,c)
        #print('FBZ', np.pi/(2*np.pi*b/L))

        pow =   L//math.gcd(2*b, L) if c == np.pi/4 else 1 
        print('transaltion pow', pow)
        #print('L =',L)
        #N = np.conj(condensate)@condensate

        A1_R = A_m_Real(J_parallel, J_perp,U, chi,condensate,chi*(1-p),0)

        A2_R = A_m_Real( J_parallel, J_perp,U,chi, condensate,-chi*p,1)
        B1_R = B_m_Real( U, condensate,0)
        B2_R = B_m_Real(U, condensate,1)
        C1_R = C_m_Real(condensate, J_perp)
        C2_R = C1_R
        A1_R_hole = A_m_Real_hole(J_parallel,J_perp, U,chi, condensate,chi*(1-p),0)
        A2_R_hole = A_m_Real_hole(J_parallel,J_perp, U, chi,condensate,-chi*p,1)


        eom_matrix = construct_L(A1_R, B1_R, A2_R, B2_R, C1_R, C2_R,A1_R_hole,A2_R_hole,L,pow,0.000001)

        eom_eigenvalues_unsorted, eom_eigenvectors_unsorted = np.linalg.eig(eom_matrix )
        sorted_indices = np.argsort(eom_eigenvalues_unsorted)


        Bogo_spec, uv_eigenvectors  = eom_eigenvalues_unsorted[sorted_indices][1+2*L:].real, eom_eigenvectors_unsorted[:, sorted_indices][:, 1+2*L:]
        W, V, renorm_density = WjmiVjmilrenormjm(condensate, uv_eigenvectors)
        H_chev = Frolich_real_space( Bogo_spec, W, V, renorm_density, R_val, L)
        eigvals, eigvecs = np.linalg.eigh(H_chev)
        eigenvalues_chev_dict[R_val] = eigvals
        eigenvectors_chev_dict[R_val] = eigvecs

        spectral_row = spec_func_ladder(Q_fixed, s, Omega_vals, eigvals, eigvecs, L, Gamma)
        spectral_data.append(spectral_row)

    spectral_data = np.array(spectral_data)  # shape: (len(R_vals), len(Omega_vals))

    R_mesh, Omega_mesh = np.meshgrid(R_vals, Omega_vals, indexing='ij')
    norm = mcolors.LogNorm(vmin=max(spectral_data.min(), 1e-8), vmax=spectral_data.max())

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(R_mesh, Omega_mesh, c=spectral_data, cmap='viridis', norm=norm, s=20)
    cbar = plt.colorbar(sc, label=r"$\mathcal{A}(\Omega, R)$")
    plt.xlabel(r"$R$", fontsize=12)
    plt.ylabel(r"$\Omega$", fontsize=12)
    plt.title(rf" $Q = {round(Q_fixed,2)}$, g={g}, U={U}, L={L}, $\chi=${round(chi,2)}", fontsize=14)
    plt.tight_layout()
    plt.show()

R_vals = np.linspace(0.1, 0.8,5)
#spectral_function_vs_R(Q_fixed=0, R_vals=R_vals, Omega_vals=Omega_vals,   s=2, g=g, L=L, Gamma=Gamma)






def spectral_function_vs_g(Q_fixed, g_vals, Omega_vals, s, eigenvalues_chev_dict, eigenvectors_chev_dict, L, Gamma):
    spectral_data = []

    for g in g_vals:
        eigenvalues_chev = eigenvalues_chev_dict[g]
        eigenvectors_chev = eigenvectors_chev_dict[g]
        spectral_row = spec_func_ladder(Q_fixed, s, Omega_vals, eigenvalues_chev, eigenvectors_chev, L, Gamma)
        spectral_data.append(spectral_row)

    spectral_data = np.array(spectral_data)  # shape: (len(g_vals), len(Omega_vals))

    G_mesh, Omega_mesh = np.meshgrid(g_vals, Omega_vals, indexing='ij')
    norm = mcolors.LogNorm(vmin=max(spectral_data.min(), 1e-8), vmax=spectral_data.max())

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(G_mesh, Omega_mesh, c=spectral_data, cmap='viridis', norm=norm, s=20)
    cbar = plt.colorbar(sc, label=r"$\mathcal{A}(\Omega, g)$")
    plt.xlabel(r"$g$", fontsize=12)
    plt.ylabel(r"$\Omega$", fontsize=12)
    plt.title(rf" $Q = {round(Q_fixed,2)}$, R={R}, U={U}, L={L}, $\chi=${round(chi,2)}", fontsize=14)
    plt.tight_layout()
    plt.show()

g_vals = np.linspace(0, 10, 100)
eigenvalues_chev_dict = {}
eigenvectors_chev_dict = {}

#for g in g_vals:
#    print('g is', g)
#    W, V, renorm_density = WjmiVjmilrenormjm(condensate, uv_eigenvectors)
#    H_chev = Frolich_real_space(condensate, Bogo_spec, W, V, renorm_density, R, L)
#    eigvals, eigvecs = np.linalg.eigh(H_chev)
#    eigenvalues_chev_dict[g] = eigvals
#    eigenvectors_chev_dict[g] = eigvecs

#spectral_function_vs_g(Q_fixed=np.pi, g_vals=g_vals, Omega_vals=Omega_vals,                      s=2, eigenvalues_chev_dict=eigenvalues_chev_dict,                     eigenvectors_chev_dict=eigenvectors_chev_dict,                     L=L, Gamma=Gamma)


s=2


#spectral_function_plot_ladder(spec_func_ladder, Q_vals,s,Omega_vals,eigenvalues_chev, eigenvectors_chev ,g,R,chi,U,L, Gamma)

#Q=0
def analyze_brightest_ladder_state(Q, Omega_min, Omega_max, s, eigenvalues_chev, eigenvectors_chev, W, V, phi, L, R, Gamma=0.25, plot=True):

    N = eigenvalues_chev.shape[0]

    j_vals = np.arange(L)
    cos_theta_Q = 1
    sin_theta_Q = 0
    phase_factors = np.exp(-1j * Q * j_vals) / np.sqrt(L)
    psi_j1 = eigenvectors_chev[np.isin(np.arange(N), (2 * np.arange(L)) * 2 * L), :]
    psi_j2 = eigenvectors_chev[np.isin(np.arange(N), (2 * np.arange(L) + 1) * 2 * L), :]
    total_amplitudes = phase_factors @ (psi_j1 * cos_theta_Q + psi_j2 * sin_theta_Q)
    Z_n = np.abs(total_amplitudes) ** 2
    energy_mask = (eigenvalues_chev >= Omega_min) & (eigenvalues_chev <= Omega_max)
    if not np.any(energy_mask):
        print(f"No states in window [{Omega_min}, {Omega_max}]")
        return None
    brightest_index = np.where(energy_mask)[0][np.argmax(Z_n[energy_mask])]
    psi_max = eigenvectors_chev[:, brightest_index]

    rho = np.zeros((L, 2))
    for j in range(L):
        for m in range(2):
            alpha = j * 2 + m
            for nu in range(2 * L):
                idx = alpha * 2 * L + nu
                rho[j, m] += np.abs(psi_max[idx]) ** 2

    rho /= np.sum(rho)

    chevy_tensor = np.zeros((L, 2, 2 * L), dtype=complex)
    for j in range(L):
        for m in range(2):
            for nu in range(2 * L):
                idx = (j * 2 + m) * (2 * L) + nu
                chevy_tensor[j, m, nu] = psi_max[idx]
                if nu != 0:
                    print('dressing part', round(np.abs(chevy_tensor[j, m, nu])**2, 3))

    g2 = np.zeros((L, 2), dtype=complex)

    for i in range(L):
        for j in range(L):
            jp = (j + i) % L
            for m in range(2):
                for p in range(2):
                    mp = m if p == 0 else 1 - m
                    phi_jp = phi[jp, mp]
                    g2[i, p] += np.abs(phi_jp) ** 2 * np.abs(chevy_tensor[j, m, 0]) ** 2
                    for nu in range(1, 2 * L):
                        g2[i, p] += np.abs(phi_jp) ** 2 * np.abs(chevy_tensor[j, m, nu]) ** 2
                        g2[i, p] += W[jp, mp, nu - 1] * np.conj(chevy_tensor[j, m, 0]) * chevy_tensor[j, m, nu]
                        g2[i, p] += np.conj(W[jp, mp, nu - 1]) * np.conj(chevy_tensor[j, m, nu]) * chevy_tensor[j, m, 0]
                        for mu in range(1, 2 * L):
                            g2[i, p] += V[jp, mp, nu - 1, mu - 1] * np.conj(chevy_tensor[j, m, mu]) * chevy_tensor[j, m, nu]
        g2[i, :] /= L

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    axs[0].plot(np.arange(L), rho[:, 1], label='Leg 2', marker='s')
    axs[0].plot(np.arange(L), rho[:, 0], label='Leg 1', marker='o', markersize=3)
    axs[0].set_title("Impurity Density")
    axs[0].set_xlabel("Site j")
    axs[0].set_ylabel(r"$\rho_{imp}$")
    axs[0].set_ylim(0, 0.03)
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(np.arange(L), g2[:, 1].real, marker='o', color='green', label='m=2')
    axs[1].plot(np.arange(L), g2[:, 0].real, marker='x', color='red', label='m=1')
    axs[1].set_title(r"$g^{(2)}_{i,m}$")
    axs[1].set_xlabel("i")
    axs[1].set_ylabel(r"$g^{(2)}$")
    axs[1].legend()
    axs[1].grid(True)

    energy_brightest = eigenvalues_chev[brightest_index]
    osc_strength = Z_n[brightest_index]
    plt.suptitle(
        f"Brightest Chevy State: Q={Q:.2f}, Ω∈[{Omega_min:.2f}, {Omega_max:.2f}], "
        f"E={energy_brightest:.4f}, Z={osc_strength:.4f},g ={g}"
    )
    plt.tight_layout()
    plt.show()


    #return { "rho": rho,  "g2": g2,"Z": Z_n[brightest_index],"Omega": eigenvalues_chev[brightest_index],"state_index": brightest_index }

##result = analyze_brightest_ladder_state(    Q=0, Omega_min=-0.2, Omega_max=5, s=2,    eigenvalues_chev=eigenvalues_chev,    eigenvectors_chev=eigenvectors_chev,    W=W, V=V, phi=renorm_density,    L=L, R=R, Gamma=0.25, plot=True)
print('should be plotted')
#print(f"Brightest state energy: {result['Omega']:.3f}")
#print(f"Oscillator strength: {result['Z']:.4f}")
#print('chevy eigenvalues', eigenvalues_chev)

def spectral_function_ladder_data(Q_vals, s, Omega_vals, eigenvalues_chev, eigenvectors_chev, g, U,L,chi,R,Gamma,time_begin_0,t0,step,n_steps,b,c):
    file_name= f"spectral_function_data_L{L}_g{g}_chi{round(chi,2)}_R{R}_U{U}_Gamma{Gamma}.csv"
    with open(file_name, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Q", "Spectral_Value"])  
        time_start = time.time()
        for Q in Q_vals:
            spectral_value = spec_func_ladder(Q, s, Omega_vals, eigenvalues_chev, eigenvectors_chev, L,Gamma)
            writer.writerow([Q,spectral_value])
            elapsed_time = time.time() - time_start
            print(f"Processed Q={Q} in {elapsed_time:.2f} seconds.")
  
        total_time = round(time.time() - time_start, 2)
    
    total_time = round(time.time() - time_start, 2)
    with open(file_name, "a", newline='') as f:
        f.write(f"# Total run time: {total_time} seconds\n")
        f.write(f"# Parameters: Gamma={Gamma}, g={g}, U={U}, R={R}, chi={chi}, L={L}, Q_range: from {Q_vals[0]} to {Q_vals[-1]}, Omega_range: from {Omega_vals[0]} to {Omega_vals[-1]}, Q_points={len(Q_vals)}, Omega_points={len(Omega_vals)}, s= {s} \n")
   



#spectral_function_ladder_data(Q_vals, s, Omega_vals, eigenvalues_chev, eigenvectors_chev, g,U, L,chi,R,Gamma,time_begin_0,t0,step,n_steps,b,c)

def spectral_function_ladder_plot_from_data(filename):
    Q_list = []
    spectral_values_list = []
    with open(filename, "r") as f:
        reader = csv.reader(f)
        header = next(reader)  
        count = 0
        for row in reader:
            if row[0].startswith("#"): 
                print('length', len(row))
                print('omega in row' , "Omega_range" in row[(7)%len(row)])
                print(row)
                if "Omega_range" in row[(7)%len(row)]:
                    print('row',row[(7)%len(row)])
                    parameters_summary = row[(7)%len(row)]
                    match = re.search(r'Omega_range: from ([\-\d\.]+) to ([\-\d\.]+)', row[(7)%len(row)])
                    if match:
                        Omega_min = float(match.group(1))
                        Omega_max = float(match.group(2))
                continue
            Q = float(row[0])
            raw_string = row[1]
            clean_values = raw_string.strip("[]").split()
            spectral_values = np.array([float(val) for val in clean_values])
            Q_list.append(Q)
            spectral_values_list.append(np.array(spectral_values))
    Q_vals = np.unique(Q_list)
    spectral_values = np.array(spectral_values_list)  
    num_Omega = spectral_values.shape[1]
    Omega_vals = np.linspace(Omega_min, Omega_max, num_Omega)
    vmin = max(spectral_values.min(), 1e-8) 
    vmax = spectral_values.max()
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    Omega_mesh, Q_mesh = np.meshgrid(Omega_vals, Q_vals, indexing='ij')

    parts = filename.replace("spectral_function_data_", "").replace(".csv", "").split("_")

    L = int(parts[0][1:])       
    g = float(parts[1][1:])       
    chi = float(parts[2][3:])    
    R = float(parts[3][1:])     
    U = float(parts[4][1:])      
    Gamma = float(parts[5][6:])   

    plt.figure(figsize=(6,5))
    plt.scatter(Q_mesh, Omega_mesh, c=spectral_values.T, cmap='viridis', norm=norm, s=40)
    #plt.colorbar(label=r"$\mathcal{A}(\Omega,Q)$")
    cbar = plt.colorbar(label=r"$\mathcal{A}(\Omega,Q)$")
    cbar.set_label(label=r"$\mathcal{A}(\Omega,Q)$", fontsize=30)
    plt.xlabel("impurity momentum Q",size =30)
    plt.ylabel(r"$\Omega$",size= 30)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.title(f" g={g}, U={U}, R={R}, L=N={L}" + r", $\chi$="+ f"{round(chi,2)},"+ r"$\Gamma=$"+ f"{Gamma}",size = 25)
    plt.show()



file_name= "spectral_function_data_L80_g0_chi1.57_R1.6_U0.2_Gamma0.25.csv"
file_name= "spectral_function_data_L80_g-5_chi1.57_R1.6_U0.2_Gamma0.25.csv"
file_name= "spectral_function_data_L80_g-5_chi1.57_R1.6_U0.2_Gamma0.25_Frolich.csv"
#file_name = "spectral_function_data_L80_g5_chi1.57_R1.6_U0.2_Gamma0.25.csv"
#file_name ="spectral_function_data_L80_g-5_chi1.57_R1.6_U0.2_Gamma0.25_MF.csv"
#file_name="spectral_function_data_L80_g-5_chi1.57_R1.05_U0.2_Gamma0.25_MF.csv"
#file_name = "spectral_function_data_L80_g-5_chi1.57_R1.05_U0.2_Gamma0.25_Frolich.csv"
#file_name ="spectral_function_data_L80_g-5_chi1.57_R1.05_U0.2_Gamma0.25.csv"
#file_name = "spectral_function_data_L80_g5_chi1.57_R1.6_U0.2_Gamma0.25_Frolich.csv"
#file_name ="spectral_function_data_L80_g5_chi1.57_R0.2_U0.2_Gamma0.25_Frolich.csv"
#file_name ="spectral_function_data_L80_g5_chi1.57_R0.2_U0.2_Gamma0.25_MF.csv"
file_name = "spectral_function_data_L20_g5_chi1.57_R0.2_U0.2_Gamma0.25_MF.csv"
#file_name = "spectral_function_data_L20_g5_chi1.57_R0.2_U0.2_Gamma0.25_MF.csv"
#file_name = "spectral_function_data_L80_g5_chi1.57_R1.05_U0.2_Gamma0.25.csv"
#file_name = "spectral_function_data_L20_g5_chi2.07_R0.25_U0.2_Gamma0.25.csv"
file_name = "spectral_function_data_L20_g-5_chi2.07_R0.25_U0.2_Gamma0.25.csv"


##spectral_function_ladder_plot_from_data(file_name)


#########################################################################################################################################################################################
########################################################################## Chevy in real space ladder : Chiral Polaron ###################################################################################
#########################################################################################################################################################################################
J_parallel = 1
J_perp = 2.2
R = J_perp/J_parallel
U = 0.2
g = 30
p = 1/2
chi = np.pi/2
L=2
N=2
Gamma = 0.25
b,c,condensate =GPE_condensate(R,chi,U,p,L,N,J_parallel, 0, 0,0, 0.05,100)
print('g,U,R,chi =', g,U,R,chi)
print("N=", N)
print('norm =', np.conj(condensate)@condensate)
print('b, omega =', b,c)
#print('FBZ', np.pi/(2*np.pi*b/L))


#print('L =',L)
#N = np.conj(condensate)@condensate

A1_R = A_m_Real(J_parallel, J_perp,U, chi,condensate,chi*(1-p),0)

A2_R = A_m_Real( J_parallel, J_perp,U,chi, condensate,-chi*p,1)
B1_R = B_m_Real( U, condensate,0)
B2_R = B_m_Real(U, condensate,1)
C1_R = C_m_Real(condensate, J_perp)
C2_R = C1_R
A1_R_hole = A_m_Real_hole(J_parallel,J_perp, U,chi, condensate,chi*(1-p),0)
A2_R_hole = A_m_Real_hole(J_parallel,J_perp, U, chi,condensate,-chi*p,1)


eom_matrix = construct_L(A1_R, B1_R, A2_R, B2_R, C1_R, C2_R,A1_R_hole,A2_R_hole,L)

eom_eigenvalues_unsorted, eom_eigenvectors_unsorted = np.linalg.eig(eom_matrix )
sorted_indices = np.argsort(eom_eigenvalues_unsorted)


Bogo_spec, uv_eigenvectors  = eom_eigenvalues_unsorted[sorted_indices][1+2*L:].real, eom_eigenvectors_unsorted[:, sorted_indices][:, 1+2*L:]


Q_vals = 2*np.pi/L*np.arange(L) - np.pi 
Omega_vals = np.linspace(-5,100,100)
time_s = time.time()
W,V,renorm_density =  WjmiVjmilrenormjm(condensate,uv_eigenvectors)
print('W V ok',W,V,renorm_density )

def Frolich_chiral_polaron(condensate, Bogo_spec, W, V, renorm_density, L, g):
    size = L*2*L  
    H_Chevy = np.zeros((size, size), dtype=complex)
    for pos in range(L):
        for pos_prime in range(L):
            same_pos = (pos == pos_prime)
            for index in range(2*L):
                for index_prime in range(2*L):
                    same_index = (index == index_prime)
                    bath = Bogo_spec[index - 1] if (same_pos and same_index and index !=0) else 0
                    impurity_kin= -1 if (same_index and (pos == (pos_prime+1)%L or pos_prime == (pos+1)%L ) ) else 0
                    chem_kin_imp = +2 if (same_pos and same_index) else 0
                    V_int = g*V[pos, 0, index - 1, index_prime - 1] if (same_pos and index != 0 and index_prime != 0) else 0
                    W_int = 0
                    if same_pos:
                        if index == 0 and index_prime != 0:
                            W_int = g * W[pos, 0, index_prime - 1]
                        elif index_prime == 0 and index != 0:
                            W_int = g * np.conj(W[pos, 0, index - 1])
                    cste = g*renorm_density[pos, 0] if (same_pos and same_index) else 0
                    H_Chevy[pos*(2*L) + index , pos_prime*(2*L) + index_prime] = cste + bath + impurity_kin + V_int + W_int + chem_kin_imp 
    return H_Chevy

print(' time to compute W V renorm density ', time.time() - time_s)
Chevy_ladder = Frolich_chiral_polaron(condensate, Bogo_spec, W, V, renorm_density, L, g)
print('time to build the chevy mat ', time.time() - time_s, Chevy_ladder.shape)
print('hermitian???', np.max(np.abs(Chevy_ladder - np.conj(Chevy_ladder).T)) )
print('diff', np.max(np.abs(Chevy_ladder - np.conj(Chevy_ladder).T)))
eigenvalues_chev, eigenvectors_chev = np.linalg.eigh(Chevy_ladder)
print('time for chevy diagonalisation', time.time() - time_s)

def spec_func_chiral(Q_vals, Omega_vals, eigenvalues_chev, eigenvectors_chev, L, R, Gamma=0.25):
    ''' for each eigenvector psi^n = psi_{0,0} psi_{0,1} psi_{0,2} psi_{0,3}...psi_{0,L} psi_{1,0} psi_{1,1}... '''
    #indices :  j n Q omega
    n_vals = np.arange(len(eigenvalues_chev))[None,:,None,None]
    j_vals = np.arange(L)[:,None,None,None]

    phase_factors = np.exp(-1j * Q_vals[None, None,:,None] * j_vals) / np.sqrt(L)  

    mask_j = np.isin(np.arange(len(eigenvalues_chev)), np.arange(L)*2*L)
    psi_j = eigenvectors_chev[mask_j, :][:,:,None,None]

    total_amplitudes = np.sum(phase_factors *psi_j, axis = 0 ) # indices n Q omega
    oscillator_strengths = np.abs(total_amplitudes) ** 2
    denom = Omega_vals[None,None,:] - eigenvalues_chev[:,None,None] + 1j*Gamma
    spectral_values = np.sum(oscillator_strengths/denom, axis = 0)
    return -2*spectral_values.imag

def plot_spectral_function_chiral(Q_vals, Omega_vals, spectral_values, g, Gamma, L, R, U, N, s, chi):
    Omega_mesh, Q_mesh = np.meshgrid(Omega_vals, Q_vals, indexing='ij')

    norm = mcolors.LogNorm(
        vmin=max(spectral_values.min(), 1e-6),
        vmax=spectral_values.max()
    )
    plt.figure(figsize=(8, 6))
    plt.scatter(Q_mesh, Omega_mesh, c=spectral_values.T, cmap='viridis', norm=norm, s=25)
    plt.colorbar(label=r"$\mathcal{A}(\Omega, Q)$")
    plt.xlabel("Impurity momentum $Q$")
    plt.ylabel(r"$\Omega$")
    plt.title(
        f"g = {g}, $\Gamma$ = {Gamma}, L = {L}, R = {R}, U = {U}, N = {round(N.real,1)}, $\chi$ = {round(chi, 2)}"
    )
    plt.show()

#print('test spec',spec_func_chiral(Q_vals, Omega_vals, eigenvalues_chev, eigenvectors_chev, L, R))

#spec_func_chiral(Q_vals, s, Omega_vals, eigenvalues_chev, eigenvectors_chev, L, R, Gamma=0.25)
#spectral_values = spec_func_chiral(Q_vals, Omega_vals, eigenvalues_chev, eigenvectors_chev, L, R,Gamma)
#plot_spectral_function_chiral(Q_vals, Omega_vals, spectral_values, g, Gamma, L, R, U, N, s, chi)

#########################################################################################################################################################################################
########################################################################## Dynamical structure factor : bath ############################################################################
#########################################################################################################################################################################################


J_parallel = 1
J_perp = 2.2
J_perp = 2
J_perp = 0.25
#J_perp = 0.1
R = J_perp/J_parallel
U = 0.2
p = 1/2
chi = np.pi*0.996
chi = np.pi*0.66
L=4*24
L= 32
L=100
L=96*2
L=20
#L=80
N=L
Gamma = 0.05
b,c,condensate =GPE_condensate(R,chi,U,p,L,N,J_parallel, 0, 0,0, 0.01,400)

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
Omega_vals = np.linspace(-0.1, 4.2,800)
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
def dyn_structure_factor_perp(q_vals, Omega_vals, m, m_prime, Bogo_spec, condensate, Bogo_eigenvectors, Gamma=0.25):
    L = len(condensate) // 2
    condensate_reshaped = condensate.reshape(L, 2)

    phi_1 = condensate_reshaped[:, 0][:, None]  
    phi_2 = condensate_reshaped[:, 1][:, None]

    u = Bogo_eigenvectors[:2*L, :].reshape(2, L, -1)
    v = np.conj(Bogo_eigenvectors[2*L:, :].reshape(2, L, -1))

    S_qw = np.zeros((len(q_vals), len(Omega_vals)), dtype=np.complex128)

    for nu in range(Bogo_spec.size):
        phase_factor = np.exp(1j * q_vals[None, :] * np.arange(L)[:, None])  

        term = (
            phi_2 * v[0, :, nu][:, None] +
            phi_1 * np.conj(u[1, :, nu])[:, None] -
            np.conj(phi_1) * v[1, :, nu][:, None] +
            phi_2 * np.conj(u[0, :, nu])[:, None]
        )  

        super_osc_strength = np.sum(phase_factor * term, axis=0)  # (len(q_vals),)

        denom = Omega_vals - Bogo_spec[nu] + 1j * Gamma  # (len(Omega_vals),)

        # ✅ broadcast correctly to (len(q_vals), len(Omega_vals))
        S_qw += (np.abs(super_osc_strength)[:, None]**2) / denom[None, :]

    return -2 / L**2 * S_qw.imag



def dyn_structure_factor_plot(f, q_vals, Omega_vals, m, m_prime, Bogo_spec, condensate, Bogo_eigenvectors, R,U,L,Gamma=0.25):
    k_values = []
    energy_values = []
    
    for i in range(2*L-1):
        k = np.angle(Bogo_eigenvectors[:,i][0]*np.conj(Bogo_eigenvectors[:,i][pow]))/pow + b*2*np.pi/L 
        trans = T_matrix_canonical_basis(pow,L)
        k= np.angle( ((trans@np.conj(Bogo_eigenvectors[:,i]) )[0]*Bogo_eigenvectors[:,i][0]) )/pow
        #k_values.append((-k + np.pi) % (2 * np.pi) ) 
        k_values.append(k )
        energy_values.append(Bogo_spec[i].real)
        
    spectral_values = f(q_vals, Omega_vals, m, m_prime, Bogo_spec, condensate, Bogo_eigenvectors, Gamma)
    #spectral_values[len(q_vals)//2,:] =0
    #spectral_values[0,:] =0
    Q_mesh, Omega_mesh = np.meshgrid(q_vals, Omega_vals, indexing='ij')
    norm = mcolors.LogNorm(vmin=max(spectral_values.min(), 2e-4), vmax=spectral_values.max() )

    #plt.figure(figsize=(7, 5))
    plt.scatter(Q_mesh.flatten(), Omega_mesh.flatten(), c=spectral_values.flatten(), cmap='viridis', norm=norm, s=40)
    plt.plot(k_values,energy_values , marker='o', linestyle='None',markersize = 3,lw=15, color='#DC143C')
    #k_values2 = [(el - np.pi + np.pi) % (2 * np.pi) - np.pi for el in k_values  ]
    #k_values3 = [(el + np.pi/2 + np.pi) % (2 * np.pi) - np.pi for el in k_values  ]
    #k_values4 = [(el + np.pi + np.pi) % (2 * np.pi) - np.pi for el in k_values  ]
    k_values2 = [(el - 2*np.pi/3) for el in k_values  ]
    k_values3 = [(el + 2*np.pi/3) for el in k_values  ]
    plt.plot(k_values2,energy_values , marker='o', linestyle='None',markersize = 3,lw=15, color='#DC143C')
    plt.plot(k_values3,energy_values , marker='o', linestyle='None',markersize = 3,lw=15, color='#DC143C')
    #plt.plot(k_values4,energy_values , marker='o', linestyle='None',markersize = 3,lw=15, color='#DC143C')

    cbar = plt.colorbar(label=r"$S(q,\Omega)$")
    cbar.ax.tick_params(labelsize=25) 
    cbar.set_label(r"$S(q,\Omega)$", fontsize=20)  

    plt.xlabel(r"Momentum $q$",size = 20)
    plt.ylabel(r"Frequency $\Omega$",size = 20)
    #plt.title(f"Dynamical Structure Factor for m={m+1}, R={R}, U={U}," + r"$\chi$= " +f"{round(chi/np.pi,2)}"+r"$\pi, \Gamma =$"+f'{Gamma}, L={L}',size = 15) 
    plt.title(  f"Dynamical Structure Factor for m={m+1}, R={R}, U={U}, "  + r"$\chi$= " + f"{round(chi/np.pi, 2)}" + r"$\pi, \Gamma =$" + f"{Gamma}, L={L}",  size=15,pad=20 )

    #plt.title(f"Dynamical Structure Factor for $R={R}, U={U},$" + r'$\chi=\frac{\pi}{2}, \Gamma =$'+f'{Gamma}, L={L}', size = 15)
    
    #plt.xlim(-1,1)
    #plt.ylim(-1,1)
    plt.xticks(size=15)
    plt.yticks(size=20)
    plt.tight_layout()
    plt.show()

S_perp = dyn_structure_factor_perp(q_vals, Omega_vals, m, m_prime, Bogo_spec, condensate, Bogo_eigenvectors, Gamma)

dyn_structure_factor_plot(
    dyn_structure_factor_perp,  # use the new transverse DSF function
    q_vals,
    Omega_vals,
    m=0,  # m and m_prime are not used in the transverse version
    m_prime=0,
    Bogo_spec=Bogo_spec,
    condensate=condensate,
    Bogo_eigenvectors=Bogo_eigenvectors,
    R=R,
    U=U,
    L=L,
    Gamma=Gamma
)





dyn_structure_factor_plot(dyn_structure_factor, q_vals, Omega_vals, m, m_prime, Bogo_spec, condensate, Bogo_eigenvectors,R,U,L,Gamma)

#dyn_structure_factor_plot(dyn_structure_factor_bonding, q_vals, Omega_vals, m, m_prime, Bogo_spec, condensate, Bogo_eigenvectors,R,U,L,Gamma)


