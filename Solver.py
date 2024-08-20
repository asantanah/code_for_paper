
from numpy import *
from qutip import *
#from qutip.qobj import Qobj, issuper, isoper

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

        # polaroid transformation
        pol_arg = (r - r.dag()) * ((ga/wr) * Na + (gb/wr) * Nb) 
        pol = (pol_arg.expm())

        # P operator
        P_arg = -(r - r.dag()) * ((ga/wr))
        P_arg_half = (P_arg / 2)
        P_half = P_arg_half.expm()

        rho_p = (pol * rho_ss * pol.dag())

        # Eq. 20
        a_ss_2 = (E_a - ga * (rho_p * a * Xr).tr() - X) / ((-ga**2 / wr) + 1j * (kappa_a/2) + 2 * chiAB * nb_ss - wa + Ohm_a)
        # Eq. 20 alternative
        a_ss_3 = (-E_a - ga * (P_half * rho_p * a * P_half * Xr).tr() - (2 * ga**2 / wr) * aada - X) / ((-ga**2 / wr) + 1j * (kappa_a/2) + 2 * chiAB * nb_ss - wa + Ohm_a)
        # Eq. 20 without X
        #a_ss_4 = ((2 * ga**2 / wr) * aada) / ((-ga**2 / wr) + 1j * (kappa_a/2) + 2 * chiAB * nb_ss - wa + Ohm_a)
        a_ss_4 = (E_a - (2 * ga**2 / wr) * aada - X) / ((-ga**2 / wr) + 1j * (kappa_a/2) + 2 * chiAB * nb_ss - wa + Ohm_a)
        # Eq. 20 açternative without X
        a_ss_5 = (E_a - ga * (P_half * rho_p * a * P_half * Xr).tr() - X) / ((-ga**2 / wr) + 1j * (kappa_a/2) + 2 * chiAB * nb_ss - wa + Ohm_a)
        #a_ss_5 = (E_a + ga * (rho_p * a * Xr).tr() + (2 * ga**2 / wr) * aada) / ((ga**2 / wr) + 1j * (kappa_a/2) - 2 * chiAB * nb_ss - wa + Ohm_a)

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

def Solver_TwoModesCoupledToMR_dispersiveQubit(N,w,kappa,T,E,g,G,proc,ohm_a_list):
    """
    This functions solves the steady-state for a system of two EM modes,
    A and B, coupled to single mechanical mode R and qubit in the dispersive regime, for different values of
    coupling strength g_a and driver detuning Delta_a = wa-Omega_a.

    """
    wa,wb,wq,wr = w
    kappa_a,kappa_b,kappa_q,gamma = kappa
    T_a,T_b,T_q,T_r = T
    E_a,E_b = E
    Ga,Gb = G
    gb = g

    n_th_a = n_thermal(wa,((sc.k*T_a)/(sc.hbar)))
    n_th_b = n_thermal(wb,((sc.k*T_b)/(sc.hbar)))
    n_th_q = n_thermal(wq,((sc.k*T_q)/(sc.hbar)))
    n_th_r = n_thermal(wr,((sc.k*T_r)/(sc.hbar)))

    a = tensor(destroy(N), qeye(N), qeye(2), qeye(N))
    b = tensor(qeye(N), destroy(N), qeye(2), qeye(N))
    r = tensor(qeye(N), qeye(N), qeye(2), destroy(N))
    sm = tensor(qeye(N), qeye(N), destroy(2), qeye(N))
    sz = tensor(qeye(N), qeye(N), sigmaz(), qeye(N))
    sx = tensor(qeye(N), qeye(N), sigmax(), qeye(N))
    I = tensor(qeye(N), qeye(N), qeye(2), qeye(N))

    Na = a.dag() * a
    Nb = b.dag() * b
    Nq = sm.dag() * sm
    Nr = r.dag() * r
    Xa = a.dag() + a
    Xb = b.dag() + b
    Xq = sm.dag() + sm
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

    ga = proc
    gb = 2 * pi * 5 * 1e6          # Fixed at 5 MHz

    etaA = ((ga**2)/wr)
    etaB = ((gb**2)/wr)
    etaAB = ((gb*ga)/wr)

    chiA = (Ga**2/(wq-wa))
    chiB = (Gb**2/(wq-wb))

    for i in range(len(ohm_a_list)):

        #Delta_a = Delta_a_list[i]
        #Delta_b = -((gb**2)/wr)

        Ohm_a = ohm_a_list[i] 
        Ohm_b = wb - etaB

        #Hamiltonian
        Ha = (wa-Ohm_a) * Na
        Hb = (wb-Ohm_b) * Nb
        Hr = wr * Nr
        Hint_a = -ga * Na * Xr
        Hint_b = -gb * Nb * Xr
        Hdrive_a = E_a * Xa
        Hdrive_b = E_b * Xb
        Hq = 0.5 * (wq) * sz

        Hqa = (Ga**2/(wq-wa)) * Na * sz 
        Hqb = (Gb**2/(wq-wb)) * Nb * sz 
                
        H = Ha + Hb + Hr + Hint_a + Hint_b + Hdrive_a + Hdrive_b + Hq + Hqa + Hqb

        c_ops = []
        rate = kappa_a * (1 + n_th_a)
        if rate > 0.0:
            c_ops.append(sqrt(rate) * a)

        rate = kappa_b * (1 + n_th_b)
        if rate > 0.0:
            c_ops.append(sqrt(rate) * b)

        rate = gamma * (1 + n_th_r)
        if rate > 0.0:
            c_ops.append(sqrt(rate) * r)

        rate = kappa_q * (1 + n_th_q)
        if rate > 0.0:
            c_ops.append(sqrt(rate) * sm)

        rate = kappa_a * n_th_a
        if rate > 0.0:
            c_ops.append(sqrt(rate) * a.dag())

        rate = kappa_b * n_th_b
        if rate > 0.0:
            c_ops.append(sqrt(rate) * b.dag())

        rate = kappa_q * n_th_q
        if rate > 0.0:
            c_ops.append(sqrt(rate) * sm.dag())

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

        # polaroid transformation
        pol_arg = (r - r.dag()) * ((ga/wr) * Na + (gb/wr) * Nb) 
        pol = (pol_arg.expm())

        # P operator
        P_arg = -(r - r.dag()) * ((ga/wr))
        P_arg_half = (P_arg / 2)
        P_half = P_arg_half.expm()

        rho_p = (pol * rho_ss * pol.dag())

        # Eq. 20
        a_ss_2 = (E_a - ga * (rho_p * a * Xr).tr() - X) / ((-ga**2 / wr) + 1j * (kappa_a/2) + 2 * etaAB * nb_ss - wa + Ohm_a)
        # Eq. 20 alternative
        a_ss_3 = (-E_a - ga * (P_half * rho_p * a * P_half * Xr).tr() - (2 * ga**2 / wr) * aada - X) / ((-ga**2 / wr) + 1j * (kappa_a/2) + 2 * etaAB * nb_ss - wa + Ohm_a)
        # Eq. 20 without X
        #a_ss_4 = ((2 * ga**2 / wr) * aada) / ((-ga**2 / wr) + 1j * (kappa_a/2) + 2 * chiAB * nb_ss - wa + Ohm_a)
        a_ss_4 = (E_a - (2 * ga**2 / wr) * aada - X) / ((-ga**2 / wr) + 1j * (kappa_a/2) + 2 * etaAB * nb_ss - wa + Ohm_a)
        # Eq. 20 açternative without X
        a_ss_5 = (E_a - ga * (P_half * rho_p * a * P_half * Xr).tr() - X) / ((-ga**2 / wr) + 1j * (kappa_a/2) + 2 * etaAB * nb_ss - wa + Ohm_a)
        #a_ss_5 = (E_a + ga * (rho_p * a * Xr).tr() + (2 * ga**2 / wr) * aada) / ((ga**2 / wr) + 1j * (kappa_a/2) - 2 * chiAB * nb_ss - wa + Ohm_a)

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


def Solver_TwoModesCoupledToMR_timeDomain(N,w,galist,E_drive,kappa,T,proc):
    """
    This functions solves the master equation for a system of two EM modes,
    A and B, coupled to single mechanical mode R, for a specific value of
    coupling strengths g_a and g_b for differnet values of external field drive.

    """
    wa,wb,wr = w
    kappa_a,kappa_b,gamma = kappa
    T_a,T_b,T_r = T
    E_a,E_b = E_drive

    tau_a = (kappa_a/(2*pi))**-1
    tau_b = (kappa_b/(2*pi))**-1
    t0 = 0
    t1 = 5 * tau_a
    tf = 5 * tau_a + 5 * tau_b

    tlist = linspace(t0,tf,200)

    Omega_a = proc

    n_th_a = n_thermal(wa,((sc.k*T_a)/(sc.hbar)))
    n_th_b = n_thermal(wb,((sc.k*T_b)/(sc.hbar)))
    n_th_r = n_thermal(wr,((sc.k*T_r)/(sc.hbar)))
    a = tensor(destroy(N), qeye(N), qeye(N))
    b = tensor(qeye(N), destroy(N), qeye(N))
    r = tensor(qeye(N), qeye(N), destroy(N))
    Na = a.dag() * a
    Nb = b.dag() * b
    Nr = r.dag() * r
    Xa = a.dag() + a
    Xb = b.dag() + b
    Xr = r.dag() + r
    Pa = a - a.dag()
    Pb = b - b.dag()

    # Field operators at t = t1
    listAux_a_t1_1 = []
    listAux_a_t1_2 = []
    listAux_a_t1_3 = []
    listAux_a_t1_4 = []
    listAux_a_t1_5 = []
    listAux_b_t1 = []
    listAux_Na_t1 = []
    listAux_Nb_t1 = []

    listAux_quadrature_Xa_t1 = []
    listAux_quadrature_Xb_t1 = []
    listAux_quadrature_Pa_t1 = []
    listAux_quadrature_Pb_t1 = []

    # Field operators at t = tf
    listAux_a_tf_1 = []
    listAux_a_tf_2 = []
    listAux_a_tf_3 = []
    listAux_a_tf_4 = []
    listAux_a_tf_5 = []
    listAux_b_tf = []
    listAux_Na_tf = []
    listAux_Nb_tf = []

    listAux_quadrature_Xa_tf = []
    listAux_quadrature_Xb_tf = []
    listAux_quadrature_Pa_tf = []
    listAux_quadrature_Pb_tf = []

    # Entanglement at t = t1
    listAux_X_t1 = []
    listAux_Xalt_t1 = []
    listAux_realX_t1 = []
    listAux_imagX_t1 = []
    listAux_negativity_t1 = []
    listAux_TrChiNN_t1 = []
    listAux_etaTrChiNN_t1 = []
    # Entanglement at t = tf
    listAux_X_tf = []
    listAux_Xalt_tf = []
    listAux_realX_tf = []
    listAux_imagX_tf = []
    listAux_negativity_tf = []
    listAux_TrChiNN_tf = []
    listAux_etaTrChiNN_tf = []
    # Populations at t = t1
    listAux_popA_Level_0_t1 = []
    listAux_popB_Level_0_t1 = []
    listAux_popA_Level_1_t1 = []
    listAux_popB_Level_1_t1 = []
    listAux_popA_Level_2_t1 = []
    listAux_popB_Level_2_t1 = []
    # Populations at t = tf
    listAux_popA_Level_0_tf = []
    listAux_popB_Level_0_tf = []
    listAux_popA_Level_1_tf = []
    listAux_popB_Level_1_tf = []
    listAux_popA_Level_2_tf = []
    listAux_popB_Level_2_tf = []

    ########## Main results ###########

    listAux_Sa_t1 = []
    listAux_Sa_tf = []
    listAux_Sb_t1 = []
    listAux_Sb_tf = []
    listAux_Delta1_t1 = []
    listAux_Delta1_tf = []
    listAux_Delta1_sep_t1 = []
    listAux_Delta1_sep_tf = []


    for i in range(len(galist)):

        
        ga = galist[i] 
        gb = ga

        etaA = ((ga**2)/wr)
        etaB = ((gb**2)/wr)
        etaAB = 2*((gb*ga)/wr)

        Omega_b = wb - etaB - etaAB

        #Hamiltonian
        Ha = (wa-Omega_a) * Na
        Hb = (wb-Omega_b) * Nb
        Hr = wr * Nr
        Hint_a = -ga * Na * Xr
        Hint_b = -gb * Nb * Xr
        Hdrive_a = E_a * Xa
        Hdrive_b = Xb
        #Hq = 0.5 * (wq) * sz
        #Hqa = Ga * (a * sm.dag() + a.dag() * sm)
        #Hqb = Gb * (b * sm.dag() + b.dag() * sm)
        #Hqa = (Ga**2/(wq-wa)) * Na * sz 
        #Hqb = (Gb**2/(wq-wb)) * Nb * sz 

        H0 = Ha + Hb + Hr + Hint_a + Hint_b + Hdrive_a #+ Hdrive_b #+ Hq + Hqa + Hqb

        def Hdrive_b_coeff(t, args):
            return E_b * (1-heaviside(t-t1,0))

        H = [H0,[Hdrive_b,Hdrive_b_coeff]]

        # collapse operators
        c_ops = []

        # Relaxations, temperature = 0 or >0

        # cavity-a relaxation
        rate = kappa_a * (1 + n_th_a)
        if rate > 0.0:
            c_ops.append(sqrt(rate) * a)
            
        # cavity-b relaxation
        rate = kappa_b * (1 + n_th_b)
        if rate > 0.0:
            c_ops.append(sqrt(rate) * b)
            
        # mechanical oscillator relaxation
        rate = gamma * (1 + n_th_r)
        if rate > 0.0:
            c_ops.append(sqrt(rate) * r)
            
        # Excitations, only temperature > 0  

        rate = kappa_a * n_th_a
        if rate > 0.0:
            c_ops.append(sqrt(rate) * a.dag())

        rate = kappa_b * n_th_b
        if rate > 0.0:
            c_ops.append(sqrt(rate) * b.dag())

        # mechanical oscillator excitation    
        rate = gamma * n_th_r
        if rate > 0.0:
            c_ops.append(sqrt(rate) * r.dag())
        
        # intial state
        rho0 = tensor(coherent_dm(N,0),coherent_dm(N,0),coherent_dm(N,0))

        # Steady-state density operators
        sol = mesolve(H,rho0,tlist,c_ops,[],options=Odeoptions(nsteps=100000))
        rho_ss_1 = sol.states[98]
        rho_ss_f = sol.states[-1]
        
        chi_ss_1 = rho_ss_1 - tensor(ptrace(rho_ss_1, (0)),
                                     ptrace(rho_ss_1, (1)), 
                                     ptrace(rho_ss_1, (2)))

        chi_ss_f = rho_ss_f - tensor(ptrace(rho_ss_f, (0)),
                                     ptrace(rho_ss_f, (1)), 
                                     ptrace(rho_ss_f, (2)))

        # Computing X and Y
        X_1 = (2 * etaAB) * expect(b.dag() * b * a, chi_ss_1)
        X_f = (2 * etaAB) * expect(b.dag() * b * a, chi_ss_f)
        X_alt_1 = expect(b.dag() * b * a, chi_ss_1)
        X_alt_f = expect(b.dag() * b * a, chi_ss_f)

        listAux_X_t1.append(abs(X_1))
        listAux_X_tf.append(abs(X_f))
        listAux_Xalt_t1.append(abs(X_alt_1))
        listAux_Xalt_tf.append(abs(X_alt_f))

        listAux_realX_t1.append(real(X_1))
        listAux_realX_tf.append(real(X_f))

        listAux_imagX_t1.append(imag(X_1))
        listAux_imagX_tf.append(imag(X_f))

        # Computing field amplitudes, method 1
        a_ss_1 = expect(a, rho_ss_1)
        b_ss_1 = expect(b, rho_ss_1)

        a_ss_f = expect(a, rho_ss_f)
        b_ss_f = expect(b, rho_ss_f)

        listAux_a_t1_1.append(a_ss_1)
        listAux_b_t1.append(b_ss_1)

        listAux_a_tf_1.append(a_ss_f)
        listAux_b_tf.append(b_ss_f)

        # Computing average number operator
        na_ss_1 = na_ss_1 = expect(Na, rho_ss_1)
        nb_ss_1 = expect(Nb, rho_ss_1)

        na_ss_f = expect(Na, rho_ss_f)
        nb_ss_f = expect(Nb, rho_ss_f)

        listAux_Na_t1.append(abs(na_ss_1))
        listAux_Sa_t1.append((kappa_a**2/E_a**2)*abs(na_ss_1))
        listAux_Nb_t1.append(abs(nb_ss_1))
        listAux_Sb_t1.append((kappa_b**2/E_b**2)*abs(nb_ss_1))

        listAux_Na_tf.append(abs(na_ss_f))
        listAux_Nb_tf.append(abs(nb_ss_f))
        # Quadrature operator X
        Xa_ss_1 = expect(Xa, rho_ss_1)
        Xb_ss_1 = expect(Xb, rho_ss_1)

        Xa_ss_f = expect(Xa, rho_ss_f)
        Xb_ss_f = expect(Xb, rho_ss_f)

        listAux_quadrature_Xa_t1.append(abs(Xa_ss_1))
        listAux_quadrature_Xb_t1.append(abs(Xb_ss_1))

        listAux_quadrature_Xa_tf.append(abs(Xa_ss_f))
        listAux_quadrature_Xb_tf.append(abs(Xb_ss_f))
        # Quadrature operator P
        Pa_ss_1 = expect(Pa, rho_ss_1)
        Pb_ss_1 = expect(Pb, rho_ss_1)

        Pa_ss_f = expect(Pa, rho_ss_f)
        Pb_ss_f = expect(Pb, rho_ss_f)

        listAux_quadrature_Pa_t1.append(abs(Pa_ss_1))
        listAux_quadrature_Pb_t1.append(abs(Pb_ss_1))

        listAux_quadrature_Pa_tf.append(abs(Pa_ss_f))
        listAux_quadrature_Pb_tf.append(abs(Pb_ss_f))
        ########################
        TrChiNN_t1 = (chi_ss_1 * Na * Nb).tr()
        etaTrChiNN_t1 = etaAB * (chi_ss_1 * Na * Nb).tr()
        delta1_1 =  etaA * expect(Na * Na, rho_ss_1) + etaAB * na_ss_1 * nb_ss_1 + etaTrChiNN_t1
        delta1_sep_1 =  etaA * expect(Na * Na, rho_ss_1) + etaAB * na_ss_1 * nb_ss_1 

        TrChiNN_tf = (chi_ss_f * Na * Nb).tr()
        etaTrChiNN_tf = etaAB * (chi_ss_f * Na * Nb).tr() 

        delta1_f =  etaA * expect(Na * Na, rho_ss_1) + etaAB * na_ss_1 * nb_ss_1 + etaTrChiNN_t1
        delta1_sep_f =  etaA * expect(Na * Na, rho_ss_1) + etaAB * na_ss_1 * nb_ss_1 

        listAux_TrChiNN_t1.append(TrChiNN_t1)
        listAux_etaTrChiNN_t1.append(etaTrChiNN_t1)
        listAux_Delta1_t1.append(delta1_1)
        listAux_Delta1_sep_t1.append(delta1_sep_1)

        listAux_TrChiNN_tf.append(TrChiNN_tf)
        listAux_etaTrChiNN_tf.append(etaTrChiNN_tf)
        listAux_Delta1_tf.append(delta1_f)
        listAux_Delta1_sep_tf.append(delta1_sep_f)

        # Computing field amplitude, method 2
        aada_1 = expect(a * a.dag() * a, rho_ss_1)
        aada_f = expect(a * a.dag() * a, rho_ss_f)

        # polaroid transformation
        pol_arg = (r - r.dag()) * ((ga/wr) * Na + (gb/wr) * Nb) 
        pol = (pol_arg.expm())

        # P operator
        P_op_a = 1j * (r - r.dag()) * (ga/wr)
        P_op_b = 1j * (r - r.dag()) * (gb/wr)

        expP_half = (1j * P_op_a/2).expm()

        rho_p_1 = (pol * rho_ss_1 * pol.dag())
        rho_p_f = (pol * rho_ss_f * pol.dag())

        # Eq. 20 (with approximation)
        a_ss_1_2 = (E_a + (2 * etaA) * aada_1 + ga * (rho_p_1 * a * Xr).tr() + X_1) / (etaA + 1j * (kappa_a/2) - 2 * etaAB * nb_ss_1 - wa + Omega_a)
        a_ss_f_2 = (E_a + (2 * etaA) * aada_f + ga * (rho_p_f * a * Xr).tr() + X_f) / (etaA + 1j * (kappa_a/2) - 2 * etaAB * nb_ss_f - wa + Omega_a)
        # Eq. 20 alternative (exact)
        a_ss_1_3 = (E_a + ga * (expP_half * rho_p_1 * a * expP_half * Xr).tr() + (2 * etaA) * aada_1 + X_1) / (etaA + 1j * (kappa_a/2) - 2 * etaAB * nb_ss_1 - wa + Omega_a)
        a_ss_f_3 = (E_a + ga * (expP_half * rho_p_f * a * expP_half * Xr).tr() + (2 * etaA) * aada_f + X_f) / (etaA + 1j * (kappa_a/2) - 2 * etaAB * nb_ss_f - wa + Omega_a)
        # Eq. 20 without X (with approximation)
        a_ss_1_4 = (E_a + (2 * etaA) * aada_1 + ga * (rho_p_1 * a * Xr).tr()) / (etaA + 1j * (kappa_a/2) - 2 * etaAB * nb_ss_1 - wa + Omega_a)
        a_ss_f_4 = (E_a + (2 * etaA) * aada_f + ga * (rho_p_f * a * Xr).tr()) / (etaA + 1j * (kappa_a/2) - 2 * etaAB * nb_ss_f - wa + Omega_a)
        # Eq. 20 alternative without X (exact)
        a_ss_1_5 = (E_a + ga * (expP_half * rho_p_1 * a * expP_half * Xr).tr() + (2 * etaA) * aada_1) / (etaA + 1j * (kappa_a/2) - 2 * etaAB * nb_ss_1 - wa + Omega_a)
        a_ss_f_5 = (E_a + ga * (expP_half * rho_p_f * a * expP_half * Xr).tr() + (2 * etaA) * aada_f) / (etaA + 1j * (kappa_a/2) - 2 * etaAB * nb_ss_f - wa + Omega_a)

        listAux_a_t1_2.append(a_ss_1_2)
        listAux_a_tf_2.append(a_ss_f_2)
        listAux_a_t1_3.append(a_ss_1_3)
        listAux_a_tf_3.append(a_ss_f_3)
        listAux_a_t1_4.append(a_ss_1_4)
        listAux_a_tf_4.append(a_ss_f_4)
        listAux_a_t1_5.append(a_ss_1_5)
        listAux_a_tf_5.append(a_ss_f_5)

        # Computing populations
        rhoA_1 = ptrace(rho_ss_1, (0))
        rhoA_f = ptrace(rho_ss_f, (0))
        rhoB_1 = ptrace(rho_ss_1, (1))
        rhoB_f = ptrace(rho_ss_f, (1))

        # Ground state
        p0A_1 = (fidelity(rhoA_1, fock(N, 0)))**2
        p0B_1 = (fidelity(rhoB_1, fock(N, 0)))**2
        p0A_f = (fidelity(rhoA_f, fock(N, 0)))**2
        p0B_f = (fidelity(rhoB_f, fock(N, 0)))**2
        # First excited state
        p1A_1 = (fidelity(rhoA_1, fock(N, 1)))**2
        p1B_1 = (fidelity(rhoB_1, fock(N, 1)))**2
        p1A_f = (fidelity(rhoA_f, fock(N, 1)))**2
        p1B_f = (fidelity(rhoB_f, fock(N, 1)))**2
        # Second excited state
        p2A_1 = (fidelity(rhoA_1, fock(N, 2)))**2
        p2B_1 = (fidelity(rhoB_1, fock(N, 2)))**2
        p2A_f = (fidelity(rhoA_1, fock(N, 2)))**2
        p2B_f = (fidelity(rhoB_f, fock(N, 2)))**2

        listAux_popA_Level_0_t1.append(p0A_1)
        listAux_popB_Level_0_t1.append(p0B_1)
        listAux_popA_Level_1_t1.append(p1A_1)
        listAux_popB_Level_1_t1.append(p1B_1)
        listAux_popA_Level_2_t1.append(p2A_1)
        listAux_popB_Level_2_t1.append(p2B_1)

        listAux_popA_Level_0_tf.append(p0A_f)
        listAux_popB_Level_0_tf.append(p0B_f)
        listAux_popA_Level_1_tf.append(p1A_f)
        listAux_popB_Level_1_tf.append(p1B_f)
        listAux_popA_Level_2_tf.append(p2A_f)
        listAux_popB_Level_2_tf.append(p2B_f)

        # Computing negativity between mode A and B
        rhoAB_1 = ptrace(rho_ss_1, (0, 1))
        rhoAB_f = ptrace(rho_ss_f, (0, 1))

        neg_1 = negativity(rhoAB_1, 0, method='eigenvalues', logarithmic=False)
        neg_f = negativity(rhoAB_f, 0, method='eigenvalues', logarithmic=False)
        
        listAux_negativity_t1.append(neg_1)
        listAux_negativity_tf.append(neg_f)
        
    # t = t1
    absA_t1_list = [abs(k) for k in listAux_a_t1_1]
    absA_t1_list_2 = [abs(k) for k in listAux_a_t1_2]
    absA_t1_list_3 = [abs(k) for k in listAux_a_t1_3]
    absA_t1_list_4 = [abs(k) for k in listAux_a_t1_4]
    absA_t1_list_5 = [abs(k) for k in listAux_a_t1_5]
    absB_t1_list = [abs(k) for k in listAux_b_t1]
    # t = tf
    absA_tf_list = [abs(k) for k in listAux_a_tf_1]
    absA_tf_list_2 = [abs(k) for k in listAux_a_tf_2]
    absA_tf_list_3 = [abs(k) for k in listAux_a_tf_3]
    absA_tf_list_4 = [abs(k) for k in listAux_a_tf_4]
    absA_tf_list_5 = [abs(k) for k in listAux_a_tf_5]
    absB_tf_list = [abs(k) for k in listAux_b_tf]

    output =[absA_t1_list,                         #0
            absA_t1_list_2,                        #1
            absA_t1_list_3,                        #2
            absA_t1_list_4,                        #3
            absA_t1_list_5,                        #4
            absB_t1_list,                          #5
            absA_tf_list,                          #6
            absA_tf_list_2,                        #7
            absA_tf_list_3,                        #8
            absA_tf_list_4,                        #9
            absA_tf_list_5,                        #10
            absB_tf_list,                          #11
            listAux_X_t1,                          #12
            listAux_X_tf,                          #13
            listAux_realX_t1,                      #14
            listAux_realX_tf,                      #15
            listAux_imagX_t1,                      #16
            listAux_imagX_tf,                      #17
            listAux_Na_t1,                         #18
            listAux_Na_tf,                         #19
            listAux_Nb_t1,                         #20
            listAux_Nb_tf,                         #21
            listAux_TrChiNN_t1,                    #22
            listAux_TrChiNN_tf,                    #23
            listAux_etaTrChiNN_t1,                 #24
            listAux_etaTrChiNN_tf,                 #25
            listAux_negativity_t1,                 #26
            listAux_negativity_tf,                 #27
            listAux_popA_Level_0_t1,               #28
            listAux_popA_Level_1_t1,               #26
            listAux_popA_Level_2_t1,               #27
            listAux_popA_Level_0_tf,               #28
            listAux_popA_Level_1_tf,               #29
            listAux_popA_Level_2_tf,               #30
            listAux_popB_Level_0_t1,               #31
            listAux_popB_Level_1_t1,               #32
            listAux_popB_Level_2_t1,               #33
            listAux_popB_Level_0_tf,               #34
            listAux_popB_Level_1_tf,               #35
            listAux_popB_Level_2_tf,               #36
            listAux_quadrature_Xa_t1,              #37 
            listAux_quadrature_Xb_t1,              #38  
            listAux_quadrature_Pa_t1,              #39
            listAux_quadrature_Pb_t1,              #40
            listAux_quadrature_Xa_tf,              #41
            listAux_quadrature_Xb_tf,              #42
            listAux_quadrature_Pa_tf,              #43
            listAux_quadrature_Pb_tf,               #44
            listAux_Sa_t1,                         #45
            listAux_Sb_t1,                         #46
            listAux_Sa_tf,                         #47
            listAux_Sb_tf,                         #48
            listAux_Delta1_t1,                     #49
            listAux_Delta1_t1,                     #51
            listAux_Delta1_sep_t1,                 #52
            listAux_Delta1_sep_tf                  #53
            ]                      

    return  output

def Solver_TwoModesCoupledToMR_dispersiveQubit(N,w,kappa,T,E,g,G,proc,ohm_a_list):
    """
    This functions solves the steady-state for a system of two EM modes,
    A and B, coupled to single mechanical mode R and qubit in the dispersive regime, for different values of
    coupling strength g_a and driver detuning Delta_a = wa-Omega_a.

    """
    wa,wb,wq,wr = w
    kappa_a,kappa_b,kappa_q,gamma = kappa
    T_a,T_b,T_q,T_r = T
    E_a,E_b = E
    Ga,Gb = G
    gb = g

    n_th_a = n_thermal(wa,((sc.k*T_a)/(sc.hbar)))
    n_th_b = n_thermal(wb,((sc.k*T_b)/(sc.hbar)))
    n_th_q = n_thermal(wq,((sc.k*T_q)/(sc.hbar)))
    n_th_r = n_thermal(wr,((sc.k*T_r)/(sc.hbar)))

    a = tensor(destroy(N), qeye(N), qeye(2), qeye(N))
    b = tensor(qeye(N), destroy(N), qeye(2), qeye(N))
    r = tensor(qeye(N), qeye(N), qeye(2), destroy(N))
    sm = tensor(qeye(N), qeye(N), destroy(2), qeye(N))
    sz = tensor(qeye(N), qeye(N), sigmaz(), qeye(N))
    sx = tensor(qeye(N), qeye(N), sigmax(), qeye(N))
    I = tensor(qeye(N), qeye(N), qeye(2), qeye(N))

    Na = a.dag() * a
    Nb = b.dag() * b
    Nq = sm.dag() * sm
    Nr = r.dag() * r
    Xa = a.dag() + a
    Xb = b.dag() + b
    Xq = sm.dag() + sm
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

    ga = proc
    gb = 2 * pi * 5 * 1e6          # Fixed at 5 MHz

    etaA = ((ga**2)/wr)
    etaB = ((gb**2)/wr)
    etaAB = ((gb*ga)/wr)

    chiA = (Ga**2/(wq-wa))
    chiB = (Gb**2/(wq-wb))

    for i in range(len(ohm_a_list)):

        #Delta_a = Delta_a_list[i]
        #Delta_b = -((gb**2)/wr)

        Ohm_a = ohm_a_list[i] 
        Ohm_b = wb - etaB

        #Hamiltonian
        Ha = (wa-Ohm_a) * Na
        Hb = (wb-Ohm_b) * Nb
        Hr = wr * Nr
        Hint_a = -ga * Na * Xr
        Hint_b = -gb * Nb * Xr
        Hdrive_a = E_a * Xa
        Hdrive_b = E_b * Xb
        Hq = 0.5 * (wq) * sz

        Hqa = (Ga**2/(wq-wa)) * Na * sz 
        Hqb = (Gb**2/(wq-wb)) * Nb * sz 
                
        H = Ha + Hb + Hr + Hint_a + Hint_b + Hdrive_a + Hdrive_b + Hq + Hqa + Hqb

        c_ops = []
        rate = kappa_a * (1 + n_th_a)
        if rate > 0.0:
            c_ops.append(sqrt(rate) * a)

        rate = kappa_b * (1 + n_th_b)
        if rate > 0.0:
            c_ops.append(sqrt(rate) * b)

        rate = gamma * (1 + n_th_r)
        if rate > 0.0:
            c_ops.append(sqrt(rate) * r)

        rate = kappa_q * (1 + n_th_q)
        if rate > 0.0:
            c_ops.append(sqrt(rate) * sm)

        rate = kappa_a * n_th_a
        if rate > 0.0:
            c_ops.append(sqrt(rate) * a.dag())

        rate = kappa_b * n_th_b
        if rate > 0.0:
            c_ops.append(sqrt(rate) * b.dag())

        rate = kappa_q * n_th_q
        if rate > 0.0:
            c_ops.append(sqrt(rate) * sm.dag())

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

        # polaroid transformation
        pol_arg = (r - r.dag()) * ((ga/wr) * Na + (gb/wr) * Nb) 
        pol = (pol_arg.expm())

        # P operator
        P_arg = -(r - r.dag()) * ((ga/wr))
        P_arg_half = (P_arg / 2)
        P_half = P_arg_half.expm()

        rho_p = (pol * rho_ss * pol.dag())

        # Eq. 20
        a_ss_2 = (E_a - ga * (rho_p * a * Xr).tr() - X) / ((-ga**2 / wr) + 1j * (kappa_a/2) + 2 * etaAB * nb_ss - wa + Ohm_a)
        # Eq. 20 alternative
        a_ss_3 = (-E_a - ga * (P_half * rho_p * a * P_half * Xr).tr() - (2 * ga**2 / wr) * aada - X) / ((-ga**2 / wr) + 1j * (kappa_a/2) + 2 * etaAB * nb_ss - wa + Ohm_a)
        # Eq. 20 without X
        #a_ss_4 = ((2 * ga**2 / wr) * aada) / ((-ga**2 / wr) + 1j * (kappa_a/2) + 2 * chiAB * nb_ss - wa + Ohm_a)
        a_ss_4 = (E_a - (2 * ga**2 / wr) * aada - X) / ((-ga**2 / wr) + 1j * (kappa_a/2) + 2 * etaAB * nb_ss - wa + Ohm_a)
        # Eq. 20 açternative without X
        a_ss_5 = (E_a - ga * (P_half * rho_p * a * P_half * Xr).tr() - X) / ((-ga**2 / wr) + 1j * (kappa_a/2) + 2 * etaAB * nb_ss - wa + Ohm_a)
        #a_ss_5 = (E_a + ga * (rho_p * a * Xr).tr() + (2 * ga**2 / wr) * aada) / ((ga**2 / wr) + 1j * (kappa_a/2) - 2 * chiAB * nb_ss - wa + Ohm_a)

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


def Solver_TwoModesCoupledToMR_steadystate(N,w,g,J,E_drive,Omega,kappa,T,proc):
    """
    This functions solves the master equation for a system of two EM modes,
    A and B, coupled to single mechanical mode R, for a specific value of
    coupling strengths g_a and g_b for differnet values of external field drive.

    """
    wa,wb,wr = w
    kappa_a,kappa_b,gamma = kappa
    T_a,T_b,T_r = T
    E_a,E_b = E_drive
    J = J

    ga, gb = g

    etaA = ((ga**2)/wr)
    etaB = ((gb**2)/wr)
    etaAB = 2 * ((gb*ga)/wr)

    Omega_a = proc   
    Omega_b = Omega

    n_th_a = n_thermal(wa,((sc.k*T_a)/(sc.hbar)))
    n_th_b = n_thermal(wb,((sc.k*T_b)/(sc.hbar)))
    n_th_r = n_thermal(wr,((sc.k*T_r)/(sc.hbar)))
    a = tensor(destroy(N), qeye(N), qeye(N))
    b = tensor(qeye(N), destroy(N), qeye(N))
    r = tensor(qeye(N), qeye(N), destroy(N))
    Na = a.dag() * a
    Nb = b.dag() * b
    Nr = r.dag() * r
    Xa = a.dag() + a
    Xb = b.dag() + b
    Xr = r.dag() + r
    Pa = a - a.dag()
    Pb = b - b.dag()

    #Hamiltonian
    Ha = (wa - Omega_a) * Na
    Hb = (wb - Omega_b) * Nb
    Hr = wr * Nr
    Hint_a = -ga * Na * Xr
    Hint_b = -gb * Nb * Xr
    Hdrive_a = 1j * E_a * (a.dag()  - a)
    Hdrive_b = 1j * E_b * (b.dag()  - b)
    Htunneling = J * (a * b.dag() + a.dag() * b)
    #Hq = 0.5 * (wq) * sz
    #Hqa = Ga * (a * sm.dag() + a.dag() * sm)
    #Hqb = Gb * (b * sm.dag() + b.dag() * sm)
    #Hqa = (Ga**2/(wq-wa)) * Na * sz 
    #Hqb = (Gb**2/(wq-wb)) * Nb * sz 



    H = Ha + Hb + Hr + Hint_a + Hint_b + Hdrive_a + Hdrive_b #+ Htunneling#+ Hq + Hqa + Hqb

    #def Hdrive_b_coeff(t, args):
    #    return E_b * (1-heaviside(t-t1,0))

    #H = [H0,[Hdrive_b,Hdrive_b_coeff]]

    # collapse operators
    c_ops = []

    # Relaxations, temperature = 0 or >0

    # cavity-a relaxation
    rate = kappa_a * (1 + n_th_a)
    if rate > 0.0:
        c_ops.append(sqrt(rate) * a)
        
    # cavity-b relaxation
    rate = kappa_b * (1 + n_th_b)
    if rate > 0.0:
        c_ops.append(sqrt(rate) * b)
        
    # mechanical oscillator relaxation
    rate = gamma * (1 + n_th_r)
    if rate > 0.0:
        c_ops.append(sqrt(rate) * r)
        
    # Excitations, only temperature > 0  

    rate = kappa_a * n_th_a
    if rate > 0.0:
        c_ops.append(sqrt(rate) * a.dag())

    rate = kappa_b * n_th_b
    if rate > 0.0:
        c_ops.append(sqrt(rate) * b.dag())

    # mechanical oscillator excitation    
    rate = gamma * n_th_r
    if rate > 0.0:
        c_ops.append(sqrt(rate) * r.dag())
    

    # Steady-state density operators
    rho_ss = steadystate(H,c_ops)
                    

    return  rho_ss