
from numpy import *
from qutip import *

from numpy.linalg import *
import multiprocessing as mp
import scipy.constants as sc
import time
import datetime
import os


def Solver_TwoModesCoupledToMR(N,wa,wb,wr,kappa_a,kappa_b,gamma,n_th_r,galist,Delta_a_list):
    """
    This functions solves the master equation for a system of

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

    List2D_FieldAmplitude_A = []
    List2D_FieldAmplitude_A_alt = []
    List2D_FieldAmplitude_B = []
    List2D_AverageNumberOp_A = []
    List2D_AverageNumberOp_B = []

    pA0_xy_list = []
    pA1_xy_list = []
    pA2_xy_list = []

    pB0_xy_list = []
    pB1_xy_list = []
    pB2_xy_list = []

    S_plus_xy_list = []
    S_minus_xy_list = []
    reS_plus_xy_list = []
    reS_minus_xy_list = []
    imS_plus_xy_list = []
    imS_minus_xy_list = []
    neg_xy_list = []

    DeltaE_xy_list = []
    T_xy_list = []

    for k in range(len(galist)):

        # Field operators
        a_list = []
        b_list = []
        Na_list = []
        Nb_list = []
        a_list_2 = []

        # Entanglement
        S_plus_list = []
        S_minus_list = []
        DeltaE_list = []
        T_list = []
        reS_plus_list = []
        reS_minus_list = []
        imS_plus_list = []
        imS_minus_list = []
        neg_list = []

        # Populations
        pA_0_list = []
        pB_0_list = []
        pA_1_list = []
        pB_1_list = []
        pA_2_list = []
        pB_2_list = []

        for i in range(len(Delta_a_list)):

            ga = galist[k]
            gb = 2 * pi * 5 * 1e6          # Fixed at 5 MHz

            chiA = ((ga**2)/wr)
            chiB = ((gb**2)/wr)
            chiAB = ((gb*ga)/wr)

            E_a = 2 * pi * 40 * 1e3
            #Ohm_a = ohm_a_list[i]
            Delta_a = Delta_a_list[i]

            E_b = 2 * pi * 40 * 1e3
            Ohm_b = wb - chiB  # 0.9999294416838251 * wb
            Delta_b = -((gb**2)/wr)

            #Ha = (wa - Ohm_a) * Na
            #Hb = (wb - Ohm_b) * Nb
            Ha = -Delta_a * Na
            Hb = -Delta_b * Nb
            Hr = wr * Nr
            Hint_a = -ga * Na * Xr
            Hint_b = -gb * Nb * Xr
            Hdrive_a = E_a * Xa
            Hdrive_b = E_b * Xb
            
            H = Ha + Hb + Hr + Hint_a + Hint_b + Hdrive_a + Hdrive_b

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

            #rate = gamma * n_th_r
            # if rate > 0.0:
            #    c_ops.append(sqrt(rate) * r.dag())

            rho_ss = steadystate(H, c_ops)
            chi_ss = rho_ss - tensor(ptrace(rho_ss, (0)),
                                    ptrace(rho_ss, (1)), ptrace(rho_ss, (2)))

            #S_plus_matrix  = chi_ss * b.dag() * b * a.dag()
            #S_minus_matrix = chi_ss * b.dag() * b * a

            #S_plus  = (2* ga * gb / wr) * S_plus_matrix.tr()
            #S_minus = (2* ga * gb / wr) * S_minus_matrix.tr()

            S_plus = (2 * ga * gb / wr) * expect(b.dag() * b * a.dag(), chi_ss)
            S_minus = (2 * ga * gb / wr) * expect(b.dag() * b * a, chi_ss)
            Tx = (ga * gb / wr) * expect(Na * Nb, chi_ss)

            T_list.append(Tx)
            S_plus_list.append(abs(S_plus))
            S_minus_list.append(abs(S_minus))

            reS_plus_list.append(real(S_plus))
            reS_minus_list.append(real(S_minus))

            imS_plus_list.append(imag(S_plus))
            imS_minus_list.append(imag(S_minus))

            a_ss = expect(a, rho_ss)
            #a_ss = (rho_ss * a).tr()
            b_ss = expect(b, rho_ss)

            aada = expect(a * a.dag() * a, rho_ss)
            #adada = expect(a.dag() * a.dag() * a, rho_ss)
            #bbdb = expect(b * b.dag() * b, rho_ss)

            pol_arg = (r.dag() - r) * ((ga/wr) * Na + (gb/wr) * Nb)
            pol = pol_arg.expm()
            #pol_arg_half = (pol_arg / 2)
            #pol_half = pol_arg_half.expm()

            na_ss = expect(Na, rho_ss)
            na2_ss = expect(Na**2, rho_ss)
            nb_ss = expect(Nb, rho_ss)

            DeltaE = -(ga**2/wr) * na2_ss - (ga * gb / wr) * na_ss * nb_ss - Tx
            DeltaE_list.append(DeltaE)

            rho_p = pol * rho_ss * pol.dag()

            a_ss_2 = (E_a + ga * (rho_p * a * Xr).tr() + (2 * ga**2 / wr) * aada + S_minus) / ((-ga**2 / wr) + 1j * (kappa_a/2) - 2 * chiAB * nb_ss - Delta_a)
            a_list_2.append(a_ss_2)

            a_list.append(a_ss)
            b_list.append(b_ss)

            Na_list.append(abs(na_ss))
            Nb_list.append(abs(nb_ss))

            rhoAB = ptrace(rho_ss, (0, 1))
            rhoA = ptrace(rhoAB, (0))
            rhoB = ptrace(rhoAB, (1))

            # Ground state
            p0A = (fidelity(rhoA, fock(N, 0)))**2
            p0B = (fidelity(rhoB, fock(N, 0)))**2
            # First excited state
            p1A = (fidelity(rhoA, fock(N, 1)))**2
            p1B = (fidelity(rhoB, fock(N, 1)))**2
            # Second excited state
            p2A = (fidelity(rhoA, fock(N, 2)))**2
            p2B = (fidelity(rhoB, fock(N, 2)))**2

            pA_0_list.append(p0A)
            pB_0_list.append(p0B)
            pA_1_list.append(p1A)
            pB_1_list.append(p1B)
            pA_2_list.append(p2A)
            pB_2_list.append(p2B)

            neg = negativity(rhoAB, 0, method='eigenvalues')
            neg_list.append(neg)

        #fig, axes = plt.subplots(1,1, figsize=(10,7))

        absA_list = [abs(k) for k in a_list]
        absA_list_2 = [abs(k) for k in a_list_2]
        absB_list = [abs(k) for k in b_list]
        #x_list = [(k-wa) / (2*pi*1e3) for k in ohm_a_list]

        #maxA_2 = max(absA_list_2)

        #absA_list_norm_2 = [k / maxA_2  for k in absA_list_2]

        #axes.plot(x_list,absA_list,label=r'$\langle \hat{a} \rangle$', lw=2.0)
        #axes.plot(x_list,absB_list,label=r'$\langle \hat{b} \rangle$', lw=2.0)
        #axes.plot(x_list,Na_list,label=r'$\vert\langle \hat{a}^\dagger\hat{a} \rangle\vert$', lw=2.0)
        #axes.plot(x_list,Nb_list,label=r'$\langle \hat{b}^\dagger\hat{b} \rangle\vert$', lw=2.0)

        #xposition = [-chiA/ (2*pi*1e3),-chiA/ (2*pi*1e3)-chiAB/ (2*pi*1e3)]
        #plt.axvline(x=xposition[0], color='blue', linestyle='--')
        #plt.axvline(x=xposition[1], color='red', linestyle='--')

        #axes.set_xlabel(r'$\omega_a - \omega_a^d$ (kHz)',rotation=0,fontsize= 20.0)
        #axes.set_ylabel(r'$\langle \hat{a}\rangle$',rotation=90,fontsize= 22.0)
        #axes.tick_params(axis='both', which='major', labelsize=16)
        #axes.tick_params(axis='both', which='minor', labelsize=16)
        # axes.legend(loc=1,fontsize=16)

        List2D_FieldAmplitude_A.append(absA_list)
        List2D_FieldAmplitude_A_alt.append(absA_list_2)
        List2D_FieldAmplitude_B.append(absB_list)
        List2D_AverageNumberOp_A.append(Na_list)
        List2D_AverageNumberOp_B.append(Nb_list)

        pA0_xy_list.append(pA_0_list)
        pA1_xy_list.append(pA_1_list)
        pA2_xy_list.append(pA_2_list)

        pB0_xy_list.append(pB_0_list)
        pB1_xy_list.append(pB_1_list)
        pB2_xy_list.append(pB_2_list)

        S_plus_xy_list.append(S_plus_list)
        S_minus_xy_list.append(S_minus_list)
        reS_plus_xy_list.append(reS_plus_list)
        reS_minus_xy_list.append(reS_minus_list)
        imS_plus_xy_list.append(imS_plus_list)
        imS_minus_xy_list.append(imS_minus_list)
        neg_xy_list.append(neg_list)

        DeltaE_xy_list.append(DeltaE_list)
        T_xy_list.append(T_list)

    return List2D_FieldAmplitude_A, List2D_FieldAmplitude_A_alt, List2D_FieldAmplitude_B, List2D_AverageNumberOp_A,List2D_AverageNumberOp_B, S_plus_xy_list , S_minus_xy_list , reS_plus_xy_list
