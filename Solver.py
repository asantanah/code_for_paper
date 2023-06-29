
from numpy import *
from qutip import *

from numpy.linalg import *
import multiprocessing as mp
import scipy.constants as sc
import time
import datetime
import os


def Solver_TwoModesCoupledToMR(N,wr,kappa_a,kappa_b,gamma,n_th_r,Ea,Eb,galist,Delta_a_list):
    """
    This functions solves the steady-state for a system of two EM modes,
    A and B, coupled to single mechanical mode R, for different values of
    coupling strength g_a and driver detuning Delta_a.

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

    list2D_FieldAmplitude_modeA = []
    list2D_FieldAmplitude_modeA_alt = []
    list2D_FieldAmplitude_modeB = []
    list2D_AverageNumberOp_modeA = []
    list2D_AverageNumberOp_modeB = []

    list2D_populationLevel0_modeA = []
    list2D_populationLevel1_modeA = []
    list2D_populationLevel2_modeA = []

    list2D_populationLevel0_modeB = []
    list2D_populationLevel1_modeB = []
    list2D_populationLevel2_modeB = []

    list2D_Y_modeA = []
    list2D_X_modeA = []
    list2D_realY_modeA = []
    list2D_realX_modeA = []
    list2D_imagY_modeA = []
    list2D_imagX_modeA = []
    list2D_Negativity_ModesAB = []

    for k in range(len(galist)):

        # Field operators
        listAux_fieldAmp_modeA_1 = []
        listAux_fieldAmp_modeA_2 = []
        listAux_fieldAmp_modeB = []
        listAux_NumberOp_modeA = []
        listAux_NumberOp_modeB = []

        # Entanglement
        listAux_Y_modeA = []
        listAux_X_modeA = []
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

        for i in range(len(Delta_a_list)):

            ga = galist[k]
            gb = 2 * pi * 5 * 1e6          # Fixed at 5 MHz

            chiA = ((ga**2)/wr)
            chiB = ((gb**2)/wr)
            chiAB = ((gb*ga)/wr)

            Delta_a = Delta_a_list[i]

            Delta_b = -((gb**2)/wr)

            #Hamiltonian
            Ha = -Delta_a * Na
            Hb = -Delta_b * Nb
            Hr = wr * Nr
            Hint_a = -ga * Na * Xr
            Hint_b = -gb * Nb * Xr
            Hdrive_a = Ea * Xa
            Hdrive_b = Eb * Xb
            
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

            listAux_Y_modeA.append(abs(Y))
            listAux_X_modeA.append(abs(X))

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
            na2_ss = expect(Na**2, rho_ss)
            nb_ss = expect(Nb, rho_ss)

            listAux_NumberOp_modeA.append(abs(na_ss))
            listAux_NumberOp_modeB.append(abs(nb_ss))

            # Computing field amplitude, method 2
            aada = expect(a * a.dag() * a, rho_ss)

            pol_arg = (r.dag() - r) * ((ga/wr) * Na + (gb/wr) * Nb)
            pol = pol_arg.expm()
            #pol_arg_half = (pol_arg / 2)
            #pol_half = pol_arg_half.expm()

            rho_p = pol * rho_ss * pol.dag()

            a_ss_2 = (Ea + ga * (rho_p * a * Xr).tr() + (2 * ga**2 / wr) * aada + X) / ((-ga**2 / wr) + 1j * (kappa_a/2) - 2 * chiAB * nb_ss - Delta_a)
            
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
            ########################################

        absA_list = [abs(k) for k in listAux_fieldAmp_modeA_1]
        absA_list_2 = [abs(k) for k in listAux_fieldAmp_modeA_2]
        absB_list = [abs(k) for k in listAux_fieldAmp_modeB]

        list2D_FieldAmplitude_modeA.append(absA_list)
        list2D_FieldAmplitude_modeA_alt.append(absA_list_2)
        list2D_FieldAmplitude_modeB.append(absB_list)
        list2D_AverageNumberOp_modeA.append(listAux_NumberOp_modeA)
        list2D_AverageNumberOp_modeB.append(listAux_NumberOp_modeB)

        list2D_populationLevel0_modeA.append(listAux_populationLevel0_modeA)
        list2D_populationLevel1_modeA.append(listAux_populationLevel1_modeA)
        list2D_populationLevel2_modeA.append(listAux_populationLevel2_modeA)

        list2D_populationLevel0_modeB.append(listAux_populationLevel0_modeB)
        list2D_populationLevel1_modeB.append(listAux_populationLevel1_modeB)
        list2D_populationLevel2_modeB.append(listAux_populationLevel2_modeB)

        list2D_Y_modeA.append(listAux_Y_modeA)
        list2D_X_modeA.append(listAux_X_modeA)
        list2D_realY_modeA.append(listAux_realY_modeA)
        list2D_realX_modeA.append(listAux_realX_modeA)
        list2D_imagY_modeA.append(listAux_imagY_modeA)
        list2D_imagX_modeA.append(listAux_imagX_modeA)
        list2D_Negativity_ModesAB.append(listAux_negativity_modesAB)

        ########################################

    return [list2D_FieldAmplitude_modeA,       #0
            list2D_FieldAmplitude_modeA_alt,   #1
            list2D_FieldAmplitude_modeB,       #2
            list2D_AverageNumberOp_modeA,      #3
            list2D_AverageNumberOp_modeB,      #4
            list2D_X_modeA,                    #5
            list2D_Y_modeA,                    #6
            list2D_realX_modeA,                #7
            list2D_realY_modeA,                #8
            list2D_imagX_modeA,                #9
            list2D_imagY_modeA,                #10
            list2D_Negativity_ModesAB,         #11
            list2D_populationLevel0_modeA,     #12
            list2D_populationLevel1_modeA,     #13
            list2D_populationLevel2_modeA,     #14
            list2D_populationLevel0_modeB,     #15
            list2D_populationLevel1_modeB,     #16
            list2D_populationLevel2_modeB]     #17
