
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
