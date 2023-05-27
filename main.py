# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import fem
from copy import deepcopy

## ----------------------------------------------------------------------------
#%% Materials properties
#
#--- Definition of the material's parameters
#
Mat = dict()
Mat['rho'] = 4.43e-9  #[ton/mm3]
Mat['E'] = 110000     #[MPa]
Mat['B'] = 770        #[MPa]
Mat['Sigma_y'] = 955  #[MPa]
Mat['n'] = 0.557      #[-]

## ----------------------------------------------------------------------------
#%% Load incrementation
#
#--- Definition of the steps
#
nStep = 10                        #[-]
tStep = 1/nStep                  #[-]

## ----------------------------------------------------------------------------
#%% Geometry data

geo = dict()
geo['L'] = 200                                #[mm]
geo['S'] = 50*20                              #[mm2]

geo['nelt'] = 100
geo['mesh'] = np.linspace(0, geo['L'], geo['nelt']+1)
geo['L_elt'] = geo['L']/(len(geo['mesh'])-1)

x_elem = []
for i in range(len(geo['mesh'])):
    x_elem.append((geo['mesh'][i] + geo['mesh'][i-1])/2)
x_elem = x_elem[1:]


## ----------------------------------------------------------------------------
#%% Loads and Boundary Condition

w_bar = 39000/60*2*np.pi                                   #[rad/s]
w = np.linspace(0,w_bar,nStep)
load = np.ones((len(w),len(geo['mesh'])))
for i in range(len(w)):
    load[i][:] = Mat['rho']*geo['S']*w[i]**2*geo['mesh']
geo['BC'] = [[0, 0]]                                       # List with [#nodes, displacement imposed]

## ----------------------------------------------------------------------------
#%% Structures historic plasticity initialisation

struc = dict()
struc['Eps_pl'] = np.zeros(geo['nelt'])
struc['sigma'] = np.zeros(geo['nelt'])
struc['p'] = np.zeros(geo['nelt'])
struc['H'] = np.zeros(geo['nelt'])
struc['R'] = np.zeros(geo['nelt'])
struc['U'] = np.zeros(len(geo['mesh']))
struc['eps'] = np.zeros(geo['nelt'])

## ----------------------------------------------------------------------------
#%% Simulation

fe = [0]*nStep
for i in range(nStep):
    print('--> Step %d (t = %.2f, w = %d tr/min)' % (i+1,(i+1)*tStep,w[i]*60/(2*np.pi)))
    geo['load'] = load[i,:]
    fe[i] = fem.FiniteElement(geo,Mat,struc)
    struc['Eps_pl'] = deepcopy(fe[i].Eps_pl)
    struc['sigma'] = deepcopy(fe[i].sigma)
    struc['p'] = deepcopy(fe[i].p)
    struc['H'] = deepcopy(fe[i].H)
    struc['R'] = deepcopy(fe[i].R)
    struc['U'] = deepcopy(fe[i].U)
    struc['eps'] = deepcopy(fe[i].eps)
## -----------------------------------------------------------------------------
#%% Analytic Solution

Sigma_analy = Mat['rho'] * w_bar**2 /2 * (geo['L']**2 - geo['mesh']**2)
U_analy = Mat['rho'] *geo['mesh']* w_bar**2 /(2*Mat['E']) * (geo['L']**2 - geo['mesh']**2/3)

## -----------------------------------------------------------------------------
#%% Plots

plt.figure()
plt.plot(x_elem, fe[nStep-1].sigma)
plt.plot(geo['mesh'], Sigma_analy)
plt.xlabel('r [mm]')
plt.ylabel('$\sigma$ [MPa]')
plt.grid()
plt.show()

plt.figure()
plt.plot(geo['mesh'], fe[nStep-1].U)
plt.plot(geo['mesh'], U_analy)
plt.xlabel('r [mm]')
plt.ylabel('u [mm]')
plt.grid()
plt.show()