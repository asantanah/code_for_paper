
from numpy import *
from qutip import *

from numpy.linalg import *
import multiprocessing as mp
import scipy.constants as sc
import time
import datetime
import os

import matplotlib.pyplot as plt



def Solver_2Modes_QNonlinearOM_rhoSS(N_a,N_b,N_r,wa,wb,wr,kappa_a,kappa_b,gamma,n_th_r,E_a,E_b,ga,gb,Delta_a,Delta_b):
    """
    This function solves for the steady-state density matrix for a system of 
    N = 2 EM modes, A and B, coupled to a single-mechanical mode R.

    It returns:
        rho_ss : QuTiP Qobj
            Steady-state density matrix of the full system A-B-R.
    """

    # Operators
    a = tensor(destroy(N_a), qeye(N_b), qeye(N_r))
    b = tensor(qeye(N_a), destroy(N_b), qeye(N_r))
    r = tensor(qeye(N_a), qeye(N_b), destroy(N_r))

    Na = a.dag() * a
    Nb = b.dag() * b
    Nr = r.dag() * r

    Xa = a.dag() + a
    Xb = b.dag() + b
    Xr = r.dag() + r

    # Hamiltonian
    Ha = -Delta_a * Na
    Hb = -Delta_b * Nb
    Hr = wr * Nr

    Hint_a = -ga * Na * Xr
    Hint_b = -gb * Nb * Xr

    Hdrive_a = E_a * Xa
    Hdrive_b = E_b * Xb

    H = Ha + Hb + Hr + Hint_a + Hint_b + Hdrive_a + Hdrive_b

    # Collapse operators
    c_ops = []

    if kappa_a > 0.0:
        c_ops.append(sqrt(kappa_a) * a)

    if kappa_b > 0.0:
        c_ops.append(sqrt(kappa_b) * b)

    rate = gamma * (1 + n_th_r)
    if rate > 0.0:
        c_ops.append(sqrt(rate) * r)

    rate = gamma * n_th_r
    if rate > 0.0:
        c_ops.append(sqrt(rate) * r.dag())

    # Steady-state density matrix
    rho_ss = steadystate(H, c_ops, method='power')

    return rho_ss

######################################################################################################
######################################################################################################
######################################################################################################


######################################################################################################
######################################################################################################
######################################################################################################

def Solver_2Modes_QNonlinearOM_voltageA(
    job_index,
    N_a, N_b, N_r,
    wa, wb, wr,
    kappa_a, kappa_b,
    gamma,
    n_th_r,
    E_a, E_b,
    ga, gb,
    Delta_a, Delta_b,
    kappa_ex_a=None,
    G_a=1.0 + 0.0j,
    phi_in_a=0.0,
    alpha_in_a=None,
    Ns=1,
    noise_sigma=0.0,
    seed=None
):
    """
    Solves rho_ss using Solver_2Modes_QNonlinearOM_rhoSS and compares:

        1) direct QuTiP field:
              <a_1>_ss = Tr[a rho_ss]

        2) reconstructed field from synthetic demodulated voltages:
              <a_1>_rec = (alpha_in_a - <V_1>/G_a) / sqrt(kappa_ex_a)

    Notes
    -----
    - Uses the input-output convention:
          a_out = a_in - sqrt(kappa_ex_a) a_1
    - kappa_ex_a defaults to kappa_a, i.e. fully external coupling.
    - G_a is the calibrated complex gain.
    - noise_sigma is the standard deviation of complex voltage noise.
      Set noise_sigma = 0 for an exact calibration test.
    """

    import numpy as np

    if kappa_ex_a is None:
        kappa_ex_a = kappa_a

    if Ns < 1:
        Ns = 1

    #=====================================================
    # Solve steady state using your existing solver
    #=====================================================
    rho_ss = Solver_2Modes_QNonlinearOM_rhoSS(
        N_a, N_b, N_r,
        wa, wb, wr,
        kappa_a, kappa_b,
        gamma,
        n_th_r,
        E_a, E_b,
        ga, gb,
        Delta_a, Delta_b
    )

    #=====================================================
    # Direct field from rho_ss
    #=====================================================
    a = tensor(destroy(N_a), qeye(N_b), qeye(N_r))
    a1_ss = expect(a, rho_ss)

    #=====================================================
    # Eq. 3.34 reconstruction for mode A
    #=====================================================

    a = tensor(destroy(N_a), qeye(N_b), qeye(N_r))
    b = tensor(qeye(N_a), destroy(N_b), qeye(N_r))
    r = tensor(qeye(N_a), qeye(N_b), destroy(N_r))

    Na = a.dag() * a
    Nb = b.dag() * b
    Xr = r.dag() + r

    # Direct field
    a1_ss = expect(a, rho_ss)

    # Photon number of mode B
    Nb_ss = expect(Nb, rho_ss)

    # Self-nonlinear moment <a_1 N_1>
    a1Na_ss = expect(a * Na, rho_ss)

    # Connected inter-mode term X_1
    chi_ss = rho_ss - tensor(
        ptrace(rho_ss, 0),
        ptrace(rho_ss, 1),
        ptrace(rho_ss, 2)
    )

    X1 = (2 * ga * gb / wr) * expect(a * Nb, chi_ss)

    # Polaron-transformed density matrix
    Up_arg = (r - r.dag()) * ((ga / wr) * Na + (gb / wr) * Nb)
    Up = Up_arg.expm()

    rho_d = Up * rho_ss * Up.dag()

    # R_1 term, using the symmetrized definition in the thesis
    Ea_half = ((ga / (2 * wr)) * (r - r.dag())).expm()

    R1 = (Ea_half * rho_d * a * Ea_half * Xr).tr()

    # Denominator using the Delta convention of your Solver.py
    den = (
        1j * kappa_a / 2
        + Delta_a
        - ga**2 / wr
        + (2 * ga * gb / wr) * Nb_ss
    )

    a1_eq334 = (
        E_a
        - ga * R1
        - (2 * ga**2 / wr) * a1Na_ss
        - X1
    ) / den
    #=====================================================
    # Synthetic input tone amplitude
    #=====================================================
    # In normalized simulation units, this choice is consistent with
    # the usual drive-amplitude relation E_a ~ sqrt(kappa_ex_a) alpha_in_a.
    #
    # If you want to use real experimental powers, replace this by:
    #
    # alpha_in_a = sqrt(P_in_a / (hbar * Omega_a)) * exp(1j * phi_in_a)
    #
    if alpha_in_a is None:
        alpha_in_a = (E_a / np.sqrt(kappa_ex_a)) * np.exp(1j * phi_in_a)

    #=====================================================
    # Generate synthetic demodulated voltage samples
    #=====================================================
    V_clean = G_a * (alpha_in_a - np.sqrt(kappa_ex_a) * a1_eq334)

    if noise_sigma > 0:
        rng = np.random.default_rng(seed)
        noise = (noise_sigma / np.sqrt(2.0)) * (
            rng.normal(size=Ns) + 1j * rng.normal(size=Ns)
        )
        V_samples = V_clean + noise
    else:
        V_samples = np.full(Ns, V_clean, dtype=complex)

    V_avg = np.mean(V_samples)

    #=====================================================
    # Reconstruct <a_1> from voltages
    #=====================================================
    a1_rec = (alpha_in_a - V_avg / G_a) / np.sqrt(kappa_ex_a)

    abs_error = abs(a1_rec - a1_ss)
    rel_error = abs_error / max(abs(a1_ss), 1e-15)

    output = [
        job_index,          # 0
        E_a,                # 1
        Delta_a,            # 2
        a1_ss,              # 3 direct complex <a_1>_ss
        a1_rec,             # 4 reconstructed complex <a_1>
        V_avg,              # 5 averaged complex voltage
        abs(a1_ss),         # 6
        abs(a1_rec),        # 7
        abs_error,          # 8
        rel_error           # 9
    ]

    return output

######################################################################################################
######################################################################################################
######################################################################################################


######################################################################################################
######################################################################################################
######################################################################################################
def Solver_TwoModesCoupledToMR_Sim1(N_a,N_b,N_r,wa,wb,wr,kappa_a,kappa_b,gamma,n_th_r,E_a,E_b,ga,Delta_a):
    """
    (Simulation 1) This functions solves the steady-state and returns a list of quantities for 
    a system of two EM modes, A and B, coupled to single mechanical mode R, for different values 
    of driver detuning Delta_a = (wa-Omega_a) (x-axis) and  coupling strength g_a (y-axis/proc).

    The output order is the following:
    0. Field amplitude mode A 
    1. Field amplitude mode B
    2. Average number operator mode A
    3. Average number operator mode B
    4. Correlation C1
    5. Correlation C2
    6. Correlation Csym
    7. Negativity between mode A and B
    8. g2 for mode A
    9. g2 for mode B
    """
    a = tensor(destroy(N_a), qeye(N_b), qeye(N_r))
    b = tensor(qeye(N_a), destroy(N_b), qeye(N_r))
    r = tensor(qeye(N_a), qeye(N_b), destroy(N_r))
    Na = a.dag() * a
    Nb = b.dag() * b
    Nr = r.dag() * r
    Xa = a.dag() + a
    Xb = b.dag() + b
    Xr = r.dag() + r

    # Field operators
    listAux_fieldAmp_modeA = []
    listAux_fieldAmp_modeB = []

    listAux_NumberOp_modeA = []
    listAux_NumberOp_modeB = []

    # Correlations
    listAux_C1 = []
    listAux_C2 = []
    listAux_Csym = []

    listAux_negativity_modesAB = []

    listAux_g2_modeA = []
    listAux_g2_modeB = []

    

    gb = 5           # Fixed at 5 MHz

    chiB = ((gb**2)/wr)

    #Ohm_a = ohm_a_list[i] 
    Delta_b = -chiB

    #Hamiltonian
    Ha = -Delta_a * Na
    Hb = -Delta_b * Nb
    Hr = wr * Nr
    Hint_a = -ga * Na * Xr
    Hint_b = -gb * Nb * Xr
    Hdrive_a = E_a * Xa
    Hdrive_b = E_b * Xb
    
    H = Ha + Hb + Hr + Hint_a + Hint_b + Hdrive_a + Hdrive_b

    # Collapse operators
    c_ops = []
    rate = kappa_a
    if rate > 0.0:
        c_ops.append(sqrt(rate) * a)

    rate = kappa_b
    if rate > 0.0:
        c_ops.append(sqrt(rate) * b)

    rate = gamma * (1 + n_th_r)
    if rate > 0.0:
        c_ops.append(sqrt(rate) * r)

    rate = gamma * n_th_r
    if rate > 0.0:
        c_ops.append(sqrt(rate) * r.dag())
    
    # Steady-state density operators
    rho_ss = steadystate(H, c_ops, method='power')
    chi_ss = rho_ss - tensor(ptrace(rho_ss, (0)),
                                ptrace(rho_ss, (1)), 
                                ptrace(rho_ss, (2)))

    # Computing correlations
    C_1 = (b.dag() * b * a * chi_ss).tr()
    C_2 = (a.dag() * a * b * chi_ss).tr()
    C_sym = sqrt(abs(C_1)**2 + abs(C_2)**2)

    listAux_C1.append(abs(C_1))
    listAux_C2.append(abs(C_2))
    listAux_Csym.append(abs(C_sym))

    # Computing field amplitudes, method 1 (qutip)
    a_ss = expect(a, rho_ss)
    b_ss = expect(b, rho_ss)

    listAux_fieldAmp_modeA.append(a_ss)
    listAux_fieldAmp_modeB.append(b_ss)

    # Computing average number operator
    na_ss = expect(Na, rho_ss)
    nb_ss = expect(Nb, rho_ss)

    listAux_NumberOp_modeA.append(abs(na_ss))
    listAux_NumberOp_modeB.append(abs(nb_ss))

    # Computing negativity between mode A and B
    rhoAB = ptrace(rho_ss, (0, 1))

    neg = negativity(rhoAB, 0, method='eigenvalues')
    
    listAux_negativity_modesAB.append(neg)
    
    # Computing g2:

    g2A = (expect(a.dag() * a.dag() * a * a, rho_ss)) / (expect(Na, rho_ss)**2)
    g2B = (expect(b.dag() * b.dag() * b * b, rho_ss)) / (expect(Nb, rho_ss)**2)

    listAux_g2_modeA.append(g2A)
    listAux_g2_modeB.append(g2B)   


    output =[abs(a_ss),                          #0
             abs(b_ss),                          #1
             abs(na_ss),                         #2
             abs(nb_ss),                         #3
             abs(C_1),                           #4
             abs(C_2),                           #5
             abs(C_sym),                         #6
             neg,                                #7
             g2A,                                #8
             g2B]                                #9                  

    return  output

######################################################################################################
######################################################################################################
######################################################################################################

def Solver_TwoModesCoupledToMR_Sim2(N_a,N_b,N_r,wa,wb,wr,kappa_a,kappa_b,gamma,n_th_r,E_a,E_b,proc,galist):
    """
    (Simulation 2) This functions solves the steady-state and returns a list of quantities for 
    a system of two EM modes, A and B, coupled to single mechanical mode R, for different values 
    of coupling strengths for cavity A, g_a (x-axis), and cavity B, g_b (y-axis/proc).

    The output order is the following:
    0. Field amplitude mode A 
    1. Field amplitude mode B
    2. Average number operator mode A
    3. Average number operator mode B
    4. Correlation C1
    5. Correlation C2
    6. Correlation Csym
    7. Negativity between mode A and B
    8. g2 for mode A
    9. g2 for mode B

    """
    a = tensor(destroy(N_a), qeye(N_b), qeye(N_r))
    b = tensor(qeye(N_a), destroy(N_b), qeye(N_r))
    r = tensor(qeye(N_a), qeye(N_b), destroy(N_r))
    Na = a.dag() * a
    Nb = b.dag() * b
    Nr = r.dag() * r
    Xa = a.dag() + a
    Xb = b.dag() + b
    Xr = r.dag() + r

    # Field operators
    listAux_fieldAmp_modeA = []
    listAux_fieldAmp_modeB = []

    listAux_NumberOp_modeA = []
    listAux_NumberOp_modeB = []

    # Correlations
    listAux_C1 = []
    listAux_C2 = []
    listAux_Csym = []

    listAux_negativity_modesAB = []

    listAux_g2_modeA = []
    listAux_g2_modeB = []

    for i in range(len(galist)):

        ga = galist[i]
        gb = proc

        chiB = ((gb**2)/wr)

        Delta_a = -4 
        Delta_b = -7 

        #Hamiltonian
        Ha = -Delta_a * Na
        Hb = -Delta_b * Nb
        Hr = wr * Nr
        Hint_a = -ga * Na * Xr
        Hint_b = -gb * Nb * Xr
        Hdrive_a = E_a * Xa
        Hdrive_b = E_b * Xb
        
        H = Ha + Hb + Hr + Hint_a + Hint_b + Hdrive_a + Hdrive_b

        # Collapse operators
        c_ops = []
        rate = kappa_a
        if rate > 0.0:
            c_ops.append(sqrt(rate) * a)

        rate = kappa_b
        if rate > 0.0:
            c_ops.append(sqrt(rate) * b)

        rate = gamma * (1 + n_th_r)
        if rate > 0.0:
            c_ops.append(sqrt(rate) * r)

        rate = gamma * n_th_r
        if rate > 0.0:
            c_ops.append(sqrt(rate) * r.dag())
        
        # Steady-state density operators
        rho_ss = steadystate(H, c_ops, method='power')
        chi_ss = rho_ss - tensor(ptrace(rho_ss, (0)),
                                 ptrace(rho_ss, (1)), 
                                 ptrace(rho_ss, (2)))

        # Computing correlations
        C_1 = (b.dag() * b * a * chi_ss).tr()
        C_2 = (a.dag() * a * b * chi_ss).tr()
        C_sym = sqrt(abs(C_1)**2 + abs(C_2)**2)

        listAux_C1.append(abs(C_1))
        listAux_C2.append(abs(C_2))
        listAux_Csym.append(abs(C_sym))

        # Computing field amplitudes, method 1 (qutip)
        a_ss = expect(a, rho_ss)
        b_ss = expect(b, rho_ss)

        listAux_fieldAmp_modeA.append(a_ss)
        listAux_fieldAmp_modeB.append(b_ss)

        # Computing average number operator
        na_ss = expect(Na, rho_ss)
        nb_ss = expect(Nb, rho_ss)

        listAux_NumberOp_modeA.append(abs(na_ss))
        listAux_NumberOp_modeB.append(abs(nb_ss))

        # Computing negativity between mode A and B
        rhoAB = ptrace(rho_ss, (0, 1))

        neg = negativity(rhoAB, 0, method='eigenvalues')
        
        listAux_negativity_modesAB.append(neg)

        # Computing g2:

        g2A = (expect(a.dag() * a.dag() * a * a, rho_ss)) / (expect(Na, rho_ss)**2)
        g2B = (expect(b.dag() * b.dag() * b * b, rho_ss)) / (expect(Nb, rho_ss)**2)

        listAux_g2_modeA.append(g2A)
        listAux_g2_modeB.append(g2B)    

    absA_list = [abs(k) for k in listAux_fieldAmp_modeA]
    absB_list = [abs(k) for k in listAux_fieldAmp_modeB]

    output =[absA_list,                         #0
            absB_list,                          #1
            listAux_NumberOp_modeA,             #2
            listAux_NumberOp_modeB,             #3
            listAux_C1,                         #4
            listAux_C2,                         #5
            listAux_Csym,                       #6
            listAux_negativity_modesAB,         #7
            listAux_g2_modeA,                   #8
            listAux_g2_modeB]                   #9

    return  output

######################################################################################################
######################################################################################################
######################################################################################################

def Solver_TwoModesCoupledToMR_Sim3(N_a,N_b,N_r,wa,wb,wr,kappa_a,kappa_b,gamma,n_th_r,E_a,E_b,proc,Delta_a_list):
    """
    (Simulation 3) This functions solves the steady-state and returns a list of quantities for 
    a system of two EM modes, A and B, coupled to single mechanical mode R, for different values 
    of driver detunings Delta_a (x-axis) and Delta_b = (wb-Omega_b) (y-axis/proc).

    The output order is the following:
    0. Field amplitude mode A 
    1. Field amplitude mode B
    2. Average number operator mode A
    3. Average number operator mode B
    4. Correlation C1
    5. Correlation C2
    6. Correlation Csym
    7. Negativity between mode A and B
    8. g2 for mode A
    9. g2 for mode B
    """
    a = tensor(destroy(N), qeye(N), qeye(N))
    b = tensor(qeye(N), destroy(N), qeye(N))
    r = tensor(qeye(N), qeye(N), destroy(N))
    Na = a.dag() * a
    Nb = b.dag() * b
    Nr = r.dag() * r
    Xa = a.dag() + a
    Xb = b.dag() + b
    Xr = r.dag() + r


    # Field operators
    listAux_fieldAmp_modeA_1 = []
    listAux_fieldAmp_modeA_2 = []
    listAux_fieldAmp_modeB = []

    listAux_NumberOp_modeA = []
    listAux_NumberOp_modeB = []

    # Correlations
    listAux_a1N2_modeA = []

    listAux_C1 = []
    listAux_C2 = []
    listAux_Csym = []

    listAux_negativity_modesAB = []

    listAux_g2_modeA = []
    listAux_g2_modeB = []

    for i in range(len(ohm_a_list)):

        ga = 2 * pi * 5 * 1e6          # Fixed at 5 MHz
        gb = 2 * pi * 5 * 1e6          # Fixed at 5 MHz

        chiA = ((ga**2)/wr)
        chiB = ((gb**2)/wr)
        chiAB = ((gb*ga)/wr)

        Ohm_a = ohm_a_list[i]
        Ohm_b = proc

        #Hamiltonian
        Ha = (wa-Ohm_a) * Na
        Hb = (wb-Ohm_b) * Nb
        Hr = wr * Nr
        Hint_a = -ga * Na * Xr
        Hint_b = -gb * Nb * Xr
        Hdrive_a = E_a * Xa
        Hdrive_b = E_b * Xb
        
        H = Ha + Hb + Hr + Hint_a + Hint_b + Hdrive_a + Hdrive_b

        # Collapse operators
        c_ops = []
        rate = kappa_a
        if rate > 0.0:
            c_ops.append(sqrt(rate) * a)

        rate = kappa_b
        if rate > 0.0:
            c_ops.append(sqrt(rate) * b)

        rate = gamma * (1 + n_th_r)
        if rate > 0.0:
            c_ops.append(sqrt(rate) * r)

        rate = gamma * n_th_r
        if rate > 0.0:
            c_ops.append(sqrt(rate) * r.dag())
        
        # Steady-state density operators
        rho_ss = steadystate(H, c_ops)
        chi_ss = rho_ss - tensor(ptrace(rho_ss, (0)),
                                 ptrace(rho_ss, (1)), 
                                 ptrace(rho_ss, (2)))

        # Computing correlations
        C_1 = (b.dag() * b * a * chi_ss).tr()
        C_2 = (a.dag() * a * b * chi_ss).tr()
        C_sym = sqrt(abs(C_1)**2 + abs(C_2)**2)

        listAux_C1.append(abs(C_1))
        listAux_C2.append(abs(C_2))
        listAux_Csym.append(abs(C_sym))

        # Computing field amplitudes, method 1 (qutip)
        a_ss = expect(a, rho_ss)
        b_ss = expect(b, rho_ss)

        listAux_fieldAmp_modeA_1.append(a_ss)
        listAux_fieldAmp_modeB.append(b_ss)

        # Computing average number operator
        na_ss = expect(Na, rho_ss)
        nb_ss = expect(Nb, rho_ss)

        listAux_NumberOp_modeA.append(abs(na_ss))
        listAux_NumberOp_modeB.append(abs(nb_ss))

        # Computing field amplitude, method 2
        aada = expect(a * a.dag() * a, rho_ss)

        pol_arg = (r.dag() - r) * ((ga/wr) * Na + (gb/wr) * Nb)
        pol = (pol_arg.expm())
        #pol_arg_half = (pol_arg / 2)
        #pol_half = pol_arg_half.expm()

        rho_p = (pol * rho_ss * pol.dag())

        # Eq. 20
        a_ss_2 = (E_a + ga * (rho_p * a * Xr).tr() - ((2 * ga**2 / wr) * aada ) - ((2 * ga*gb / wr) * C_1)) / ((ga**2 / wr) + 1j * (kappa_a/2) + (2 * ga*gb / wr) * nb_ss - wa + Ohm_a)

        listAux_fieldAmp_modeA_2.append(a_ss_2)


        # Computing negativity between mode A and B
        rhoAB = ptrace(rho_ss, (0, 1))

        neg = negativity(rhoAB, 0, method='eigenvalues')
        
        listAux_negativity_modesAB.append(neg)

        # Computing g2:

        g2A = (expect(a.dag() * a.dag() * a * a, rho_ss)) / (expect(Na, rho_ss)**2)
        g2B = (expect(b.dag() * b.dag() * b * b, rho_ss)) / (expect(Nb, rho_ss)**2)

        listAux_g2_modeA.append(g2A)
        listAux_g2_modeB.append(g2B)   

    absA_list = [abs(k) for k in listAux_fieldAmp_modeA_1]
    absA_list_2 = [abs(k) for k in listAux_fieldAmp_modeA_2]

    absB_list = [abs(k) for k in listAux_fieldAmp_modeB]

    output =[absA_list,                         #0
            absA_list_2,                        #1
            absB_list,                          #2
            listAux_NumberOp_modeA,             #3
            listAux_NumberOp_modeB,             #4
            listAux_C1,                         #5
            listAux_C2,                         #6
            listAux_Csym,                       #7
            listAux_negativity_modesAB,         #8
            listAux_g2_modeA,                   #9
            listAux_g2_modeB]                   #10

    return  output

######################################################################################################
######################################################################################################
######################################################################################################

def Solver_TwoModesCoupledToMR_Sim4(N,wa,wb,wr,kappa_a,kappa_b,gamma,n_th_r,E_a,E_b,proc,ohm_a_list,Ohm_b):
    """
    This functions solves the steady-state for a system of two EM modes,
    A and B, coupled to single mechanical mode R, for different values of
    coupling strength g_a and driver detuning Delta_a = wa-Omega_a.

    """
    a = tensor(destroy(N), qeye(N), qeye(N))
    b = tensor(qeye(N), destroy(N), qeye(N))
    r = tensor(qeye(N), qeye(N), destroy(N))
    Na = a.dag() * a
    Nb = b.dag() * b
    Nr = r.dag() * r
    Xa = a.dag() + a
    Xb = b.dag() + b
    Xr = r.dag() + r


    # Field operators
    listAux_fieldAmp_modeA_1 = []
    listAux_fieldAmp_modeA_2 = []
    listAux_fieldAmp_modeB = []

    listAux_NumberOp_modeA = []
    listAux_NumberOp_modeB = []

    # Populations
    listAux_populationLevel0_modeA = []
    listAux_populationLevel0_modeB = []
    listAux_populationLevel1_modeA = []
    listAux_populationLevel1_modeB = []

    listAux_populationLevel2_modeA = []
    listAux_populationLevel2_modeB = []
    listAux_populationLevel3_modeA = []
    listAux_populationLevel3_modeB = []
    listAux_populationLevel4_modeA = []
    listAux_populationLevel4_modeB = []

    listAux_g2_modeA = []
    listAux_g2_modeB = []

    for i in range(len(ohm_a_list)):

        ga = proc 
        #gb = ga
        gb = 2 * pi * 5 * 1e6          # Fixed at 5 MHz
        
        chiA = ((ga**2)/wr)
        chiB = ((gb**2)/wr)
        chiAB = ((gb*ga)/wr)

        Ohm_a = ohm_a_list[i] 
        #Ohm_b = Ohm_a - wa + wb

        #Hamiltonian
        Ha = (wa-Ohm_a) * Na
        Hb = (wb-Ohm_b) * Nb
        Hr = wr * Nr
        Hint_a = -ga * Na * Xr
        Hint_b = -gb * Nb * Xr
        Hdrive_a = E_a * Xa
        Hdrive_b = E_b * Xb
        
        H = Ha + Hb + Hr + Hint_a + Hint_b + Hdrive_a + Hdrive_b

        # Collapse operators
        c_ops = []
        rate = kappa_a
        if rate > 0.0:
            c_ops.append(sqrt(rate) * a)

        rate = kappa_b
        if rate > 0.0:
            c_ops.append(sqrt(rate) * b)

        rate = gamma * (1 + n_th_r)
        if rate > 0.0:
            c_ops.append(sqrt(rate) * r)

        rate = gamma * n_th_r
        if rate > 0.0:
            c_ops.append(sqrt(rate) * r.dag())
        
        # Steady-state density operators
        rho_ss = steadystate(H, c_ops)
        chi_ss = rho_ss - tensor(ptrace(rho_ss, (0)),
                                 ptrace(rho_ss, (1)), 
                                 ptrace(rho_ss, (2)))


        # Computing field amplitudes, method 1 (qutip)
        a_ss = expect(a, rho_ss)
        b_ss = expect(b, rho_ss)

        listAux_fieldAmp_modeA_1.append(a_ss)
        listAux_fieldAmp_modeB.append(b_ss)

        # Computing correlations
        C_1 = (b.dag() * b * a * chi_ss).tr()

        # Computing average number operator
        na_ss = expect(Na, rho_ss)
        nb_ss = expect(Nb, rho_ss)

        listAux_NumberOp_modeA.append(abs(na_ss))
        listAux_NumberOp_modeB.append(abs(nb_ss))

        # Computing field amplitude, method 2
        aada = expect(a * a.dag() * a, rho_ss)

        pol_arg = (r.dag() - r) * ((ga/wr) * Na + (gb/wr) * Nb)
        pol = (pol_arg.expm())
        #pol_arg_half = (pol_arg / 2)
        #pol_half = pol_arg_half.expm()

        rho_p = (pol * rho_ss * pol.dag())

        # Eq. 20
        a_ss_2 = (E_a + ga * (rho_p * a * Xr).tr() - ((2 * ga**2 / wr) * aada ) - ((2 * ga*gb / wr) * C_1)) / ((ga**2 / wr) + 1j * (kappa_a/2) + (2 * ga*gb / wr) * nb_ss - wa + Ohm_a)

        listAux_fieldAmp_modeA_2.append(a_ss_2)

        # Computing populations
        rhoA = ptrace(rho_ss, (0))
        rhoB = ptrace(rho_ss, (1))

        # Ground state
        p0A = (fidelity(rhoA, fock(N, 0)))**2
        p0B = (fidelity(rhoB, fock(N, 0)))**2
        # First excited state
        p1A = (fidelity(rhoA, fock(N, 1)))**2
        p1B = (fidelity(rhoB, fock(N, 1)))**2
        if N>2:
            # Second excited state
            p2A = (fidelity(rhoA, fock(N, 2)))**2
            p2B = (fidelity(rhoB, fock(N, 2)))**2
        if N>3:
            # Third excited state
            p3A = (fidelity(rhoA, fock(N, 3)))**2
            p3B = (fidelity(rhoB, fock(N, 3)))**2
        if N>4:
            # Fourth excited state
            p4A = (fidelity(rhoA, fock(N, 4)))**2
            p4B = (fidelity(rhoB, fock(N, 4)))**2

        listAux_populationLevel0_modeA.append(p0A)
        listAux_populationLevel0_modeB.append(p0B)
        listAux_populationLevel1_modeA.append(p1A)
        listAux_populationLevel1_modeB.append(p1B)
        if N>2:
            listAux_populationLevel2_modeA.append(p2A)
            listAux_populationLevel2_modeB.append(p2B)
        if N>3:
            listAux_populationLevel3_modeA.append(p3A)
            listAux_populationLevel3_modeB.append(p3B)
        if N>4:
            listAux_populationLevel4_modeA.append(p4A)
            listAux_populationLevel4_modeB.append(p4B)

        # Computing g2:

        g2A = (expect(a.dag() * a.dag() * a * a, rho_ss)) / (expect(Na, rho_ss)**2)
        g2B = (expect(b.dag() * b.dag() * b * b, rho_ss)) / (expect(Nb, rho_ss)**2)

        listAux_g2_modeA.append(g2A)
        listAux_g2_modeB.append(g2B)    

    absA_list = [abs(k) for k in listAux_fieldAmp_modeA_1]
    absA_list_2 = [abs(k) for k in listAux_fieldAmp_modeA_2]

    absB_list = [abs(k) for k in listAux_fieldAmp_modeB]

    if N==2:
        output =[absA_list,                         #0
                absA_list_2,                        #1
                absB_list,                          #2
                listAux_NumberOp_modeA,             #3
                listAux_NumberOp_modeB,             #4
                listAux_populationLevel0_modeA,     #5
                listAux_populationLevel1_modeA,     #6
                listAux_populationLevel0_modeB,     #7
                listAux_populationLevel1_modeB,     #8
                listAux_g2_modeA,                   #9
                listAux_g2_modeB]                   #10
    elif N==3:
        output =[absA_list,                         #0
                absA_list_2,                        #1
                absB_list,                          #2
                listAux_NumberOp_modeA,             #3
                listAux_NumberOp_modeB,             #4
                listAux_populationLevel0_modeA,     #5
                listAux_populationLevel1_modeA,     #6
                listAux_populationLevel2_modeA,     #7   
                listAux_populationLevel0_modeB,     #8
                listAux_populationLevel1_modeB,     #9
                listAux_populationLevel2_modeB,     #10
                listAux_g2_modeA,                   #11
                listAux_g2_modeB]                   #12
    elif N==4:
        output =[absA_list,                         #0
                absA_list_2,                        #1
                absB_list,                          #2
                listAux_NumberOp_modeA,             #3
                listAux_NumberOp_modeB,             #4
                listAux_populationLevel0_modeA,     #5
                listAux_populationLevel1_modeA,     #6
                listAux_populationLevel2_modeA,     #7
                listAux_populationLevel3_modeA,     #8
                listAux_populationLevel0_modeB,     #9
                listAux_populationLevel1_modeB,     #10
                listAux_populationLevel2_modeB,     #11
                listAux_populationLevel3_modeB,     #12
                listAux_g2_modeA,                   #13
                listAux_g2_modeB]                   #14
        
    elif N==5:
        output =[absA_list,                         #0
                absA_list_2,                        #1
                absB_list,                          #2
                listAux_NumberOp_modeA,             #3
                listAux_NumberOp_modeB,             #4
                listAux_populationLevel0_modeA,     #5
                listAux_populationLevel1_modeA,     #6
                listAux_populationLevel2_modeA,     #7
                listAux_populationLevel3_modeA,     #8
                listAux_populationLevel4_modeA,     #9
                listAux_populationLevel0_modeB,     #10
                listAux_populationLevel1_modeB,     #11
                listAux_populationLevel2_modeB,     #12
                listAux_populationLevel3_modeB,     #13
                listAux_populationLevel4_modeB,     #14
                listAux_g2_modeA,                   #15
                listAux_g2_modeB]                   #16

    return  output

######################################################################################################
######################################################################################################
######################################################################################################


def Solver_ThreeModesCoupledToMR_Sim1(N,wa,wb,wc,wr,kappa_a,kappa_b,kappa_c,gamma,n_th_r,E_a,E_b,E_c,proc,ohm_a_list):
    """
    This functions solves the steady-state for a system of two EM modes,
    A and B, coupled to single mechanical mode R, for different values of
    coupling strength g_a and driver detuning Delta_a = wa-Omega_a.

    """
    a = tensor(destroy(N), qeye(N), qeye(N), qeye(N))
    b = tensor(qeye(N), destroy(N), qeye(N), qeye(N))
    c = tensor(qeye(N), qeye(N), destroy(N), qeye(N))
    r = tensor(qeye(N), qeye(N), qeye(N), destroy(N))
    Na = a.dag() * a
    Nb = b.dag() * b
    Nc = c.dag() * c
    Nr = r.dag() * r
    Xa = a.dag() + a
    Xb = b.dag() + b
    Xc = c.dag() + c
    Xr = r.dag() + r


    # Field operators
    listAux_fieldAmp_modeA = []
    listAux_fieldAmp_modeB = []
    listAux_fieldAmp_modeC = []

    listAux_NumberOp_modeA = []
    listAux_NumberOp_modeB = []
    listAux_NumberOp_modeC = []

    # Correlations
    listAux_a1N2_modeA = []

    listAux_C12 = []
    listAux_C21 = []
    listAux_C13 = []
    listAux_C31 = []
    listAux_Csym_1 = []
    listAux_Csym_2 = []

    listAux_negativity_modesAB = []
    listAux_negativity_modesAC = []
    listAux_negativity_modesBC = []

    for i in range(len(ohm_a_list)):

        ga = proc
        gb = 2 * pi * 5 * 1e6          # Fixed at 5 MHz
        gc = 2 * pi * 5 * 1e6          # Fixed at 5 MHz

        chiA = ((ga**2)/wr)
        chiB = ((gb**2)/wr)
        chiC = ((gc**2)/wr)

        chiAB = ((gb*ga)/wr)
        chiAC = ((gc*ga)/wr)
        chiBC = ((gc*gb)/wr)

        Ohm_a = ohm_a_list[i] 
        Ohm_b = wb - chiB
        Ohm_c = wc - chiC

        #Hamiltonian
        Ha = (wa-Ohm_a) * Na
        Hb = (wb-Ohm_b) * Nb
        Hc = (wc-Ohm_c) * Nc
        Hr = wr * Nr
        Hint_a = -ga * Na * Xr
        Hint_b = -gb * Nb * Xr
        Hint_c = -gc * Nc * Xr
        Hdrive_a = E_a * Xa
        Hdrive_b = E_b * Xb
        Hdrive_c = E_c * Xc

        H = Ha + Hb + Hc + Hr + Hint_a + Hint_b + Hint_c + Hdrive_a + Hdrive_b + Hdrive_c

        # Collapse operators
        c_ops = []
        rate = kappa_a
        if rate > 0.0:
            c_ops.append(sqrt(rate) * a)

        rate = kappa_b
        if rate > 0.0:
            c_ops.append(sqrt(rate) * b)

        rate = kappa_c
        if rate > 0.0:
            c_ops.append(sqrt(rate) * c)

        rate = gamma * (1 + n_th_r)
        if rate > 0.0:
            c_ops.append(sqrt(rate) * r)

        rate = gamma * n_th_r
        if rate > 0.0:
            c_ops.append(sqrt(rate) * r.dag())
        
        # Steady-state density operators
        rho_ss = steadystate(H, c_ops)
        chi_ss = rho_ss - tensor(ptrace(rho_ss, (0)),
                                 ptrace(rho_ss, (1)), 
                                 ptrace(rho_ss, (2)), 
                                 ptrace(rho_ss, (3)))

        # Computing correlations
        C_12 = (b.dag() * b * a * chi_ss).tr()
        C_21 = (a.dag() * a * b * chi_ss).tr()
        C_13 = (c.dag() * c * a * chi_ss).tr()
        C_31 = (a.dag() * a * c * chi_ss).tr()
        C_sym_1 = sqrt(abs(C_12)**2 + abs(C_21)**2)
        C_sym_2 = sqrt(abs(C_13)**2 + abs(C_31)**2)

        listAux_C12.append(abs(C_12))
        listAux_C21.append(abs(C_21))
        listAux_C13.append(abs(C_13))
        listAux_C31.append(abs(C_31))
        listAux_Csym_1.append(abs(C_sym_1))
        listAux_Csym_2.append(abs(C_sym_2))

        # Computing field amplitudes, method 1 (qutip)
        a_ss = expect(a, rho_ss)
        b_ss = expect(b, rho_ss)
        c_ss = expect(c, rho_ss)

        listAux_fieldAmp_modeA.append(a_ss)
        listAux_fieldAmp_modeB.append(b_ss)
        listAux_fieldAmp_modeC.append(c_ss)

        # Computing average number operator
        na_ss = expect(Na, rho_ss)
        nb_ss = expect(Nb, rho_ss)
        nc_ss = expect(Nc, rho_ss)

        listAux_NumberOp_modeA.append(abs(na_ss))
        listAux_NumberOp_modeB.append(abs(nb_ss))
        listAux_NumberOp_modeC.append(abs(nc_ss))

        # Computing negativity between mode A and B
        rhoAB = ptrace(rho_ss, (0, 1))
        rhoAC = ptrace(rho_ss, (0, 2))
        rhoBC = ptrace(rho_ss, (1, 2))

        neg_AB = negativity(rhoAB, 0, method='eigenvalues')
        neg_AC = negativity(rhoAC, 0, method='eigenvalues')
        neg_BC = negativity(rhoBC, 0, method='eigenvalues')

        listAux_negativity_modesAB.append(neg_AB)
        listAux_negativity_modesAC.append(neg_AC)
        listAux_negativity_modesBC.append(neg_BC)


    absA_list = [abs(k) for k in listAux_fieldAmp_modeA]
    absB_list = [abs(k) for k in listAux_fieldAmp_modeB]
    absC_list = [abs(k) for k in listAux_fieldAmp_modeC]

    output =[absA_list,                         #0
            absB_list,                          #1
            absC_list,                          #2
            listAux_NumberOp_modeA,             #3
            listAux_NumberOp_modeB,             #4
            listAux_NumberOp_modeC,             #5
            listAux_C12,                        #6
            listAux_C21,                        #7
            listAux_C13,                        #8
            listAux_C31,                        #9
            listAux_Csym_1,                     #10
            listAux_Csym_2,                     #11
            listAux_negativity_modesAB,         #12
            listAux_negativity_modesAC,         #13
            listAux_negativity_modesBC          #14
            ]   

    return  output

######################################################################################################
######################################################################################################
######################################################################################################

def Solver_TwoModesCoupledToMR_ExtField(N,wa,wb,wr,kappa_a,kappa_b,gamma,n_th_r,ga,gb,proc,E_b,ohm_a_list):
    """
    This functions solves the steady-state for a system of two EM modes,
    A and B, coupled to single mechanical mode R, for different values of
     external field amplitude E_a and driver detuning Delta_a = wa-Omega_a.

    """
    a = tensor(destroy(N), qeye(N), qeye(N))
    b = tensor(qeye(N), destroy(N), qeye(N))
    r = tensor(qeye(N), qeye(N), destroy(N))
    Na = a.dag() * a
    Nb = b.dag() * b
    Nr = r.dag() * r
    Xa = a.dag() + a
    Xb = b.dag() + b
    Xr = r.dag() + r


    # Field operators
    listAux_fieldAmp_modeA_1 = []
    listAux_fieldAmp_modeA_2 = []
    listAux_fieldAmp_modeA_3 = []
    listAux_fieldAmp_modeA_4 = []
    listAux_fieldAmp_modeA_5 = []
    listAux_fieldAmp_modeB = []
    listAux_NumberOp_modeA = []
    listAux_NumberOp_modeB = []

    # Entanglement
    listAux_Y_modeA = []
    listAux_X_modeA = []
    listAux_Xalt_modeA = []
    listAux_realY_modeA = []
    listAux_realX_modeA = []
    listAux_imagY_modeA = []
    listAux_imagX_modeA = []
    listAux_negativity_modesAB = []

    # Populations
    listAux_populationLevel0_modeA = []
    listAux_populationLevel0_modeB = []
    listAux_populationLevel1_modeA = []
    listAux_populationLevel1_modeB = []
    listAux_populationLevel2_modeA = []
    listAux_populationLevel2_modeB = []

    for i in range(len(ohm_a_list)):

        chiA = ((ga**2)/wr)
        chiB = ((gb**2)/wr)
        chiAB = ((gb*ga)/wr)

        #Delta_a = Delta_a_list[i]

        #Delta_b = -((gb**2)/wr)

        Ohm_a = ohm_a_list[i] 
        Ohm_b = wb - chiB

        #Hamiltonian
        Ha = (wa-Ohm_a) * Na
        Hb = (wb-Ohm_b) * Nb
        Hr = wr * Nr
        Hint_a = -ga * Na * Xr
        Hint_b = -gb * Nb * Xr
        Hdrive_a = proc * Xa
        Hdrive_b = E_b * Xb
        
        H = Ha + Hb + Hr + Hint_a + Hint_b + Hdrive_a + Hdrive_b

        # Collapse operators
        c_ops = []
        rate = kappa_a
        if rate > 0.0:
            c_ops.append(sqrt(rate) * a)

        rate = kappa_b
        if rate > 0.0:
            c_ops.append(sqrt(rate) * b)

        rate = gamma * (1 + n_th_r)
        if rate > 0.0:
            c_ops.append(sqrt(rate) * r)

        rate = gamma * n_th_r
        if rate > 0.0:
            c_ops.append(sqrt(rate) * r.dag())
        
        # Steady-state density operators
        rho_ss = steadystate(H, c_ops)
        chi_ss = rho_ss - tensor(ptrace(rho_ss, (0)),
                                    ptrace(rho_ss, (1)), 
                                    ptrace(rho_ss, (2)))

        # Computing X and Y
        Y = (2 * ga * gb / wr) * expect(b.dag() * b * a.dag(), chi_ss)
        X = (2 * ga * gb / wr) * expect(b.dag() * b * a, chi_ss)
        X_alt = expect(b.dag() * b * a, chi_ss)

        listAux_Y_modeA.append(abs(Y))
        listAux_X_modeA.append(abs(X))
        listAux_Xalt_modeA.append(abs(X_alt))

        listAux_realY_modeA.append(real(Y))
        listAux_realX_modeA.append(real(X))

        listAux_imagY_modeA.append(imag(Y))
        listAux_imagX_modeA.append(imag(X))

        # Computing field amplitudes, method 1
        a_ss = expect(a, rho_ss)
        b_ss = expect(b, rho_ss)

        listAux_fieldAmp_modeA_1.append(a_ss)
        listAux_fieldAmp_modeB.append(b_ss)

        # Computing average number operator
        na_ss = expect(Na, rho_ss)
        nb_ss = expect(Nb, rho_ss)

        listAux_NumberOp_modeA.append(abs(na_ss))
        listAux_NumberOp_modeB.append(abs(nb_ss))

        # Computing field amplitude, method 2
        aada = expect(a * a.dag() * a, rho_ss)

        pol_arg = (r - r.dag()) * ((ga/wr) * Na + (gb/wr) * Nb)
        pol = (pol_arg.expm())
        #pol_arg_half = (pol_arg / 2)
        #pol_half = pol_arg_half.expm()

        rho_p = (pol * rho_ss * pol.dag())

        # Eq. 20
        a_ss_2 = (proc - ga * (rho_p * a * Xr).tr() - (2 * ga**2 / wr) * aada - X) / ((-ga**2 / wr) + 1j * (kappa_a/2) + 2 * chiAB * nb_ss - wa + Ohm_a)
        # Eq. 20 alternative
        a_ss_3 = (proc + ga * (rho_p * a * Xr).tr() + (2 * ga**2 / wr) * aada + X) / ((ga**2 / wr) + 1j * (kappa_a/2) - 2 * chiAB * nb_ss - wa + Ohm_a)
        # Eq. 20 without X
        a_ss_4 = (proc - ga * (rho_p * a * Xr).tr() - (2 * ga**2 / wr) * aada) / ((-ga**2 / wr) + 1j * (kappa_a/2) + 2 * chiAB * nb_ss - wa + Ohm_a)
        # Eq. 20 açternative without X
        a_ss_5 = (proc + ga * (rho_p * a * Xr).tr() + (2 * ga**2 / wr) * aada) / ((ga**2 / wr) + 1j * (kappa_a/2) - 2 * chiAB * nb_ss - wa + Ohm_a)

        listAux_fieldAmp_modeA_2.append(a_ss_2)
        listAux_fieldAmp_modeA_3.append(a_ss_3)
        listAux_fieldAmp_modeA_4.append(a_ss_4)
        listAux_fieldAmp_modeA_5.append(a_ss_5)

        # Computing populations
        rhoA = ptrace(rho_ss, (0))
        rhoB = ptrace(rho_ss, (1))

        # Ground state
        p0A = (fidelity(rhoA, fock(N, 0)))**2
        p0B = (fidelity(rhoB, fock(N, 0)))**2
        # First excited state
        p1A = (fidelity(rhoA, fock(N, 1)))**2
        p1B = (fidelity(rhoB, fock(N, 1)))**2
        # Second excited state
        p2A = (fidelity(rhoA, fock(N, 2)))**2
        p2B = (fidelity(rhoB, fock(N, 2)))**2

        listAux_populationLevel0_modeA.append(p0A)
        listAux_populationLevel0_modeB.append(p0B)
        listAux_populationLevel1_modeA.append(p1A)
        listAux_populationLevel1_modeB.append(p1B)
        listAux_populationLevel2_modeA.append(p2A)
        listAux_populationLevel2_modeB.append(p2B)

        # Computing negativity between mode A and B
        rhoAB = ptrace(rho_ss, (0, 1))

        neg = negativity(rhoAB, 0, method='eigenvalues')
        
        listAux_negativity_modesAB.append(neg)

    absA_list = [abs(k) for k in listAux_fieldAmp_modeA_1]
    absA_list_2 = [abs(k) for k in listAux_fieldAmp_modeA_2]
    absA_list_3 = [abs(k) for k in listAux_fieldAmp_modeA_3]
    absA_list_4 = [abs(k) for k in listAux_fieldAmp_modeA_4]
    absA_list_5 = [abs(k) for k in listAux_fieldAmp_modeA_5]
    absB_list = [abs(k) for k in listAux_fieldAmp_modeB]

    output =[absA_list,                         #0
            absA_list_2,                        #1
            absA_list_3,                        #2
            absA_list_4,                        #3
            absA_list_5,                        #4
            absB_list,                          #5
            listAux_NumberOp_modeA,             #6
            listAux_NumberOp_modeB,             #7
            listAux_X_modeA,                    #8
            listAux_Y_modeA,                    #9
            listAux_realX_modeA,                #10
            listAux_realY_modeA,                #11
            listAux_imagX_modeA,                #12
            listAux_imagY_modeA,                #13
            listAux_negativity_modesAB,         #14
            listAux_populationLevel0_modeA,     #15
            listAux_populationLevel1_modeA,     #16
            listAux_populationLevel2_modeA,     #17
            listAux_populationLevel0_modeB,     #18
            listAux_populationLevel1_modeB,     #19
            listAux_populationLevel2_modeB,     #20
            listAux_Xalt_modeA]                 #21

    return  output

######################################################################################################
######################################################################################################
######################################################################################################

def Solver_TwoModesCoupledToMR_Detuning(N,wa,wb,wr,kappa_a,kappa_b,gamma,n_th_r,ga,gb,E_a,E_b,proc,ohm_a_list):
    """
    This functions solves the steady-state for a system of two EM modes,
    A and B, coupled to single mechanical mode R, for different values of
    driver detunings Delta_a = wa-Omega_a and Delta_b = wb-Omega_b.

    """
    a = tensor(destroy(N), qeye(N), qeye(N))
    b = tensor(qeye(N), destroy(N), qeye(N))
    r = tensor(qeye(N), qeye(N), destroy(N))
    Na = a.dag() * a
    Nb = b.dag() * b
    Nr = r.dag() * r
    Xa = a.dag() + a
    Xb = b.dag() + b
    Xr = r.dag() + r


    # Field operators
    listAux_fieldAmp_modeA_1 = []
    listAux_fieldAmp_modeA_2 = []
    listAux_fieldAmp_modeA_3 = []
    listAux_fieldAmp_modeA_4 = []
    listAux_fieldAmp_modeA_5 = []
    listAux_fieldAmp_modeB = []
    listAux_NumberOp_modeA = []
    listAux_NumberOp_modeB = []

    # Entanglement
    listAux_Y_modeA = []
    listAux_X_modeA = []
    listAux_Xalt_modeA = []
    listAux_realY_modeA = []
    listAux_realX_modeA = []
    listAux_imagY_modeA = []
    listAux_imagX_modeA = []
    listAux_negativity_modesAB = []

    # Populations
    listAux_populationLevel0_modeA = []
    listAux_populationLevel0_modeB = []
    listAux_populationLevel1_modeA = []
    listAux_populationLevel1_modeB = []
    listAux_populationLevel2_modeA = []
    listAux_populationLevel2_modeB = []

    for i in range(len(ohm_a_list)):

        chiA = ((ga**2)/wr)
        chiB = ((gb**2)/wr)
        chiAB = ((gb*ga)/wr)

        #Delta_a = Delta_a_list[i]

        #Delta_b = -((gb**2)/wr)

        Ohm_a = ohm_a_list[i] 
        Ohm_b = proc

        #Hamiltonian
        Ha = (wa-Ohm_a) * Na
        Hb = (wb-Ohm_b) * Nb
        Hr = wr * Nr
        Hint_a = -ga * Na * Xr
        Hint_b = -gb * Nb * Xr
        Hdrive_a = E_a * Xa
        Hdrive_b = E_b * Xb
        
        H = Ha + Hb + Hr + Hint_a + Hint_b + Hdrive_a + Hdrive_b

        # Collapse operators
        c_ops = []
        rate = kappa_a
        if rate > 0.0:
            c_ops.append(sqrt(rate) * a)

        rate = kappa_b
        if rate > 0.0:
            c_ops.append(sqrt(rate) * b)

        rate = gamma * (1 + n_th_r)
        if rate > 0.0:
            c_ops.append(sqrt(rate) * r)

        rate = gamma * n_th_r
        if rate > 0.0:
            c_ops.append(sqrt(rate) * r.dag())
        
        # Steady-state density operators
        rho_ss = steadystate(H, c_ops)
        chi_ss = rho_ss - tensor(ptrace(rho_ss, (0)),
                                    ptrace(rho_ss, (1)), 
                                    ptrace(rho_ss, (2)))

        # Computing X and Y
        Y = (2 * ga * gb / wr) * expect(b.dag() * b * a.dag(), chi_ss)
        X = (2 * ga * gb / wr) * expect(b.dag() * b * a, chi_ss)
        X_alt = expect(b.dag() * b * a, chi_ss)

        listAux_Y_modeA.append(abs(Y))
        listAux_X_modeA.append(abs(X))
        listAux_Xalt_modeA.append(abs(X_alt))

        listAux_realY_modeA.append(real(Y))
        listAux_realX_modeA.append(real(X))

        listAux_imagY_modeA.append(imag(Y))
        listAux_imagX_modeA.append(imag(X))

        # Computing field amplitudes, method 1
        a_ss = expect(a, rho_ss)
        b_ss = expect(b, rho_ss)

        listAux_fieldAmp_modeA_1.append(a_ss)
        listAux_fieldAmp_modeB.append(b_ss)

        # Computing average number operator
        na_ss = expect(Na, rho_ss)
        nb_ss = expect(Nb, rho_ss)

        listAux_NumberOp_modeA.append(abs(na_ss))
        listAux_NumberOp_modeB.append(abs(nb_ss))

        # Computing field amplitude, method 2
        aada = expect(a * a.dag() * a, rho_ss)

        pol_arg = (r - r.dag()) * ((ga/wr) * Na + (gb/wr) * Nb)
        pol = (pol_arg.expm())
        #pol_arg_half = (pol_arg / 2)
        #pol_half = pol_arg_half.expm()

        rho_p = (pol * rho_ss * pol.dag())

        # Eq. 20
        a_ss_2 = (E_a - ga * (rho_p * a * Xr).tr() - (2 * ga**2 / wr) * aada - X) / ((-ga**2 / wr) + 1j * (kappa_a/2) + 2 * chiAB * nb_ss - wa + Ohm_a)
        # Eq. 20 alternative
        a_ss_3 = (E_a + ga * (rho_p * a * Xr).tr() + (2 * ga**2 / wr) * aada + X) / ((ga**2 / wr) + 1j * (kappa_a/2) - 2 * chiAB * nb_ss - wa + Ohm_a)
        # Eq. 20 without X
        a_ss_4 = (E_a - ga * (rho_p * a * Xr).tr() - (2 * ga**2 / wr) * aada) / ((-ga**2 / wr) + 1j * (kappa_a/2) + 2 * chiAB * nb_ss - wa + Ohm_a)
        # Eq. 20 açternative without X
        a_ss_5 = (E_a + ga * (rho_p * a * Xr).tr() + (2 * ga**2 / wr) * aada) / ((ga**2 / wr) + 1j * (kappa_a/2) - 2 * chiAB * nb_ss - wa + Ohm_a)

        listAux_fieldAmp_modeA_2.append(a_ss_2)
        listAux_fieldAmp_modeA_3.append(a_ss_3)
        listAux_fieldAmp_modeA_4.append(a_ss_4)
        listAux_fieldAmp_modeA_5.append(a_ss_5)

        # Computing populations
        rhoA = ptrace(rho_ss, (0))
        rhoB = ptrace(rho_ss, (1))

        # Ground state
        p0A = (fidelity(rhoA, fock(N, 0)))**2
        p0B = (fidelity(rhoB, fock(N, 0)))**2
        # First excited state
        p1A = (fidelity(rhoA, fock(N, 1)))**2
        p1B = (fidelity(rhoB, fock(N, 1)))**2
        # Second excited state
        p2A = (fidelity(rhoA, fock(N, 2)))**2
        p2B = (fidelity(rhoB, fock(N, 2)))**2

        listAux_populationLevel0_modeA.append(p0A)
        listAux_populationLevel0_modeB.append(p0B)
        listAux_populationLevel1_modeA.append(p1A)
        listAux_populationLevel1_modeB.append(p1B)
        listAux_populationLevel2_modeA.append(p2A)
        listAux_populationLevel2_modeB.append(p2B)

        # Computing negativity between mode A and B
        rhoAB = ptrace(rho_ss, (0, 1))

        neg = negativity(rhoAB, 0, method='eigenvalues')
        
        listAux_negativity_modesAB.append(neg)

    absA_list = [abs(k) for k in listAux_fieldAmp_modeA_1]
    absA_list_2 = [abs(k) for k in listAux_fieldAmp_modeA_2]
    absA_list_3 = [abs(k) for k in listAux_fieldAmp_modeA_3]
    absA_list_4 = [abs(k) for k in listAux_fieldAmp_modeA_4]
    absA_list_5 = [abs(k) for k in listAux_fieldAmp_modeA_5]
    absB_list = [abs(k) for k in listAux_fieldAmp_modeB]

    output =[absA_list,                         #0
            absA_list_2,                        #1
            absA_list_3,                        #2
            absA_list_4,                        #3
            absA_list_5,                        #4
            absB_list,                          #5
            listAux_NumberOp_modeA,             #6
            listAux_NumberOp_modeB,             #7
            listAux_X_modeA,                    #8
            listAux_Y_modeA,                    #9
            listAux_realX_modeA,                #10
            listAux_realY_modeA,                #11
            listAux_imagX_modeA,                #12
            listAux_imagY_modeA,                #13
            listAux_negativity_modesAB,         #14
            listAux_populationLevel0_modeA,     #15
            listAux_populationLevel1_modeA,     #16
            listAux_populationLevel2_modeA,     #17
            listAux_populationLevel0_modeB,     #18
            listAux_populationLevel1_modeB,     #19
            listAux_populationLevel2_modeB,     #20
            listAux_Xalt_modeA]                 #21

    return  output

