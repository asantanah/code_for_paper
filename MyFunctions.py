# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 14:30:10 2014

@author: rouxinol
"""

import pylab
from numpy import *
from collections import Counter
from scipy import interpolate
from scipy import stats
import copy
from scipy.optimize import curve_fit
from io import StringIO


#_______________________________________________________


def find_nearest(array, value):
    """ Function to find the nearest 
    value in a array and return the value and index
    """
    idx = (abs(array - value)).argmin()
    print

    return array[idx], idx


#_________________________________________________
# Import data Setup Measurement


def convertQutipTotxt(file, lin=100, col=100):
    '''
    convert file from .qu format (complex) to a txt format
    to load the file use: a = loadtxt(file_path , dtype=np.complex128)
    '''
#     file = '/Users/rouxinol/Dropbox/Electromechanical Systems/DATA/Simulation_Plots/V65NEW'
    data = qload(file)
    data2 = asarray(data)
#     lin = 100
#     col = 100
    tr_c = reshape(data2[:,0],(-1,lin) )
    savetxt(file, tr_c, delimiter=' ',fmt='%.10e%+.10ej '*col) 
    a = loadtxt(file2 , dtype=np.complex128)



def ImportXYZZ(file_name, col_x=1, col_y=2, col_Amp=3, col_Phase=4):
    """
    Import file from the actual directory and divide in coluum X, Y and Matrix for Amp and Phase
    First coluum # 0

    """

    data = genfromtxt(file_name, delimiter=(","))

    # Columm definitions

#     col_x = 1
#     col_y = 2
#     col_Amp = 3
#     col_Phase = 4
    #X = []
    #Y = []

    X, X_index = unique(data[:, col_x],return_index=True)
    Y, Y_index = unique(data[:, col_y],return_index=True)
    # Check if unique function return the data in the correct order: first to last line
    if X_index.shape[0] > 1: # check if there is only one value
        if X_index[0]>X_index[1]:
            X = X[::-1]
        if Y_index[0]>Y_index[1]:
            Y = Y[::-1]

    X_len = len(unique(data[:, col_x]))
    Y_len = len(unique(data[:, col_y]))
    # print(X_len,Y_len)
    Amp = reshape(data[:, col_Amp], (-1, Y_len))
    Phase = reshape(data[:, col_Phase], (-1, Y_len))

    return X, Y, Amp, Phase


def SuperImportXYZZ(file_name,col_i=0, col_x=1, col_y=2, col_Amp=3, col_Phase=4):
    """
    Import file from the actual directory and divide in coluum X, Y and Matrix for Amp and Phase; uses col_i to divide the data.

    First coluum # 0
    col_1, col_X, col_Y, col_Amp, col_Phase
    0
    """

    data = genfromtxt(file_name, delimiter=(","))

    # Columm definitions

#     col_Phase = 4
    #X = []
    #Y = []

    # Columm definitions
    # col_i = 0
    # col_x = 1
    # col_y = 2
    # col_Amp = 3
    # col_Phase = 4
    #X = []
    #Y = []

    i,i_index = unique(data[:,col_i],return_index=True)
    X,X_index = unique(data[:, col_x],return_index=True)
    Y, Y_index = unique(data[:, col_y],return_index=True)
    # Check if unique function return the data in the correct order: first to last line
    if X_index.shape[0] > 1: # check if there is only one value
        if X_index[0]>X_index[1]:
            X = X[::-1]
        if Y_index[0]>Y_index[1]:
            Y = Y[::-1]
    i_len = len(unique(data[:, col_i]))
    X_len = len(unique(data[:, col_x]))
    Y_len = len(unique(data[:, col_y]))

    Amp = reshape(data[:, col_Amp], (i_len, Y_len,-1))
    Phase = reshape(data[:, col_Phase], (i_len, Y_len,-1))
    return X, Y, Amp, Phase

#__________________________________________________

def LineUp(File_Name, X, Y, Center=2, Divide=2, Ref_Curve=0):
    """  Line up data from File_Name (2D matrix) in relation to the Ref_Curve 
      Center: number dividing the length of X Axis "
      Divide: number dividing the length for max line up"
    """
    File = copy.deepcopy(File_Name)
    #Ref_Curve = 1
    # lst2=copy.deepcopy(lst1)
    X_len = len(X)
    Y_len = len(Y)
    List = []
#     print(Y_len,File.shapte)
    for k in range(X_len):

        List = (correlate(File[
                Ref_Curve, :] - mean(File[Ref_Curve, :]), File[k, :] - mean(File[k, :]), mode='same'))
        max_val = max(List)
    #     print(max_val)
    #     print(List)
        pos = [i for i, x in enumerate(List) if x == max_val][0]
    #     print(pos)
        pos = int(pos - X_len / Center)

    #     print(pos)
    #     print(pos)
    #     print(File[k,:])
        if pos < Y_len / Divide:
            File[k, :] = (roll(File[k, :], pos))
    #         print(pos)
    #     else:
    #         print(k)

        # print(File[k,:])

    return File

# Original Program -  History
#File = copy.deepcopy(Phase)
#Ref_Curve = 35
# lst2=copy.deepcopy(lst1)
#List = []
#Y_len = len(Y)
#X_len = len(X)
# print(Y_len,File.shape)
# for k in range(X_len):
#
#    List = (correlate(File[Ref_Curve,:] - mean(File[Ref_Curve,:]),File[k,:] - mean(File[k,:]),mode='same'))
#    max_val = max(List)
# print(max_val)
# print(List)
#    pos = [i for i, x in enumerate(List) if x == max_val][0]
# print(pos)
#    pos = int(pos - X_len )
#
# print(pos)
# print(pos)
# print(File[k,:])
#    if pos < Y_len/2:
#        File[k,:] = (roll(File[k,:],pos))
# print(pos)
# else:
# print(k)
#
#
# print(File[k,:])
#
# File.shape
#________________________________________________________


#_______________________________________________________
# Import S21 File parameters S21 amp and phase


def ImportS2P(file_name, Cols=(0, 3), Skip_head = 5, Delimiter = "\t"):
    """  Import from s2p file freq, S21 - Amp and Phase, Return Freq, S21_Amp and S21
         sometimes need to use 
         np.genfromtxt(path, delimiter=("\t"), skip_header=5, usecols=(0,3,4),dtype=None)#,invalid_raise = False)
    """
    data = genfromtxt(file_name, delimiter=(
        Delimiter), skip_header=Skip_head, usecols=(Cols[0], Cols[1]))
    freq = data[:, 0]
    S21_Amp = data[:, 1]
    #S21_Phase = data[:, 2]
    return freq, S21_Amp

#______________________________________________________________

# Program to get file path using GUI dialog box




#__________________________________________________
# PColor fix range
def PColor(Data):
    """ Function to correct the range of X and Y axis in a pcolor map
    MATLAB pcolor() always discards the last row and column of C, but matplotlib displays the last row and column if X and Y are not specified, or if X and Y have one more row and column than C."
    """
    size_X = Data.size
    step_X = (Data[0]-Data[1])

    X = array([(Data[i]+step_X/2) for i in range(size_X)])
    return append(X,X[-1]-step_X)




# Wrap phase________________________________________
def unwrap_d(Phase):
    """
    Unwrap Phase in Dregree"""
    return unwrap(Phase * pi / 180)


def wrap(Phase):
    """ Wrap back the phase 
    """
    return (Phase + pi) % (2 * pi) - pi


def wrap1(Phase):
    """ Wrap back the phase using arctan2
    """
    return arctan2(sin(Phase), cos(Phase))


# Convert dBm to V _______________________________


def dBmtoV(Amp):
    """Convert dBm to Volts
    Input Amplitude in dBm
    """
    c = 10 * log10(20)
    return pow(10, (Amp - c) / 20)


def VtodBm(V):
    """Convert Volts to dBm
    Input Voltage in Volts
    """
    c = 10 * log10(20)
    return 20 * log10(V) + c


# Convert dBm to W

def dBmtoW(Amp):
    """Convert dBm to Watts
    Imput Amp in dBm
    """
    return pow(10, (Amp - 30) / 10)


#_______________________________________________________
