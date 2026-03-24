
from numpy import *
from qutip import *

from numpy.linalg import *
import multiprocessing as mp
import scipy.constants as sc
import time
import datetime
import os


def Solver_TwoModesCoupledToMR(N,wa,wb,wr,kappa_a,kappa_b,gamma,n_th_r,E_a,E_b,proc,ohm_a_list):
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
    listAux_fieldAmp_modeA_3 = []
    listAux_fieldAmp_modeA_4 = []
    listAux_fieldAmp_modeA_5 = []
    listAux_fieldAmp_modeB = []
    listAux_NumberOp_modeA = []
    listAux_NumberOp_modeB = []

    # Entanglement
    listAux_C1 = []
    listAux_C2 = []
    listAux_Csym = []
    listAux_X_modeA = []
    listAux_realX_modeA = []
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

        ga = proc
        gb = 2 * pi * 5 * 1e6          # Fixed at 5 MHz

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
        C_1 = expect(b.dag() * b * a, chi_ss)
        C_2 = expect(a.dag() * a * b, chi_ss)
        C_sym = sqrt(abs(C_1)**2 + abs(C_2)**2)

        X = (2 * ga * gb / wr) * C_1

        listAux_X_modeA.append(abs(X))
        listAux_realX_modeA.append(real(X))
        listAux_imagX_modeA.append(imag(X))

        listAux_C1.append(abs(C_1))
        listAux_C2.append(abs(C_2))
        listAux_Csym.append(abs(C_sym))

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

        pol_arg = (r.dag() - r) * ((ga/wr) * Na + (gb/wr) * Nb)
        pol = (pol_arg.expm())
        #pol_arg_half = (pol_arg / 2)
        #pol_half = pol_arg_half.expm()

        rho_p = (pol * rho_ss * pol.dag())

        # Eq. 20
        a_ss_2 = (E_a + ga * (rho_p * a * Xr).tr() - (2 * ga**2 / wr) * aada - X) / ((ga**2 / wr) + 1j * (kappa_a/2) + 2 * chiAB * nb_ss - wa + Ohm_a)

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

    absB_list = [abs(k) for k in listAux_fieldAmp_modeB]

    output =[absA_list,                         #0
            absA_list_2,                        #1
            absB_list,                          #2
            listAux_NumberOp_modeA,             #3
            listAux_NumberOp_modeB,             #4
            listAux_X_modeA,                    #5
            listAux_C1,                         #6
            listAux_C2,                         #7
            listAux_Csym,                       #8
            listAux_realX_modeA,                #9
            listAux_imagX_modeA,                #10
            listAux_negativity_modesAB,         #11
            listAux_populationLevel0_modeA,     #12
            listAux_populationLevel1_modeA,     #13
            listAux_populationLevel2_modeA,     #14
            listAux_populationLevel0_modeB,     #15
            listAux_populationLevel1_modeB,     #16
            listAux_populationLevel2_modeB]     #17

    return  output

######################################################################################################
######################################################################################################
######################################################################################################

def Solver_TwoModesCoupledToMR_Sim1(N,wa,wb,wr,kappa_a,kappa_b,gamma,n_th_r,E_a,E_b,proc,ohm_a_list):
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

    # Correlations
    listAux_a1N2_modeA = []

    listAux_C1 = []
    listAux_C2 = []
    listAux_Csym = []

    listAux_negativity_modesAB = []

    for i in range(len(ohm_a_list)):

        ga = proc
        gb = 2 * pi * 5 * 1e6          # Fixed at 5 MHz

        chiA = ((ga**2)/wr)
        chiB = ((gb**2)/wr)
        chiAB = ((gb*ga)/wr)

        Ohm_a = ohm_a_list[i] 
        Ohm_b = wb - chiB

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
            listAux_negativity_modesAB]         #8

    return  output

######################################################################################################
######################################################################################################
######################################################################################################

def Solver_TwoModesCoupledToMR_Sim2(N,wa,wb,wr,kappa_a,kappa_b,gamma,n_th_r,E_a,E_b,proc,galist):
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

    # Correlations
    listAux_a1N2_modeA = []

    listAux_C1 = []
    listAux_C2 = []
    listAux_Csym = []

    listAux_negativity_modesAB = []

    for i in range(len(galist)):

        ga = galist[i]
        gb = proc

        chiA = ((ga**2)/wr)
        chiB = ((gb**2)/wr)
        chiAB = ((gb*ga)/wr)

        Ohm_a = wa - chiA
        Ohm_b = wb - chiB

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
            listAux_negativity_modesAB]         #8

    return  output

######################################################################################################
######################################################################################################
######################################################################################################

def Solver_TwoModesCoupledToMR_Sim3(N,wa,wb,wr,kappa_a,kappa_b,gamma,n_th_r,E_a,E_b,proc,ohm_a_list):
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

    # Correlations
    listAux_a1N2_modeA = []

    listAux_C1 = []
    listAux_C2 = []
    listAux_Csym = []

    listAux_negativity_modesAB = []

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
            listAux_negativity_modesAB]         #8

    return  output

######################################################################################################
######################################################################################################
######################################################################################################

def Solver_TwoModesCoupledToMR_Sim4(N,wa,wb,wr,kappa_a,kappa_b,gamma,n_th_r,E_a,E_b,proc,ohm_a_list):
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

    for i in range(len(ohm_a_list)):

        ga = proc         
        gb = 2 * pi * 5 * 1e6          # Fixed at 5 MHz

        chiA = ((ga**2)/wr)
        chiB = ((gb**2)/wr)
        chiAB = ((gb*ga)/wr)

        Ohm_a = ohm_a_list[i] 
        Ohm_b = wb - chiB

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
                listAux_populationLevel1_modeB]     #8   
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
                listAux_populationLevel2_modeB]     #10
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
                listAux_populationLevel3_modeB]     #12
        
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
                listAux_populationLevel4_modeB]     #14

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

