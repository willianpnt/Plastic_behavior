# -*- coding: utf-8 -*

import Vector1D as V1
import Matrix1D as M1
import numpy as np
import numpy.linalg as la
import BehaviorLaw as BL
from copy import deepcopy

class FiniteElement:
    def __init__(self, geo, Prop, struc):
        x_nodes = geo['mesh']
        nNodes = len(geo['mesh'])
        load = geo['load']
        S = geo['S']
        BC = geo['BC']
        
        
        self.L_elt = geo['L_elt']
        self.Eps_pl= deepcopy(struc['Eps_pl'])
        self.sigma= deepcopy(struc['sigma'])
        self.p= deepcopy(struc['p'][:])
        self.H = deepcopy(struc['H'])
        self.R = deepcopy(struc['R'])
        self.U = deepcopy(struc['U'])
        
        #
        #--- Iteraction Parameters
        #
        tol = 1e-3
        itermax = 10
        #
        #--- Vector Forces
        #
        self.f_ext = V1.Linear_form(x_nodes, 0, load, [])
        K = Prop['E']*S*M1.Matrix1D(x_nodes, 1, 1, [], [])
        self.f_int = V1.Linear_form(x_nodes, 1, [], self.sigma*S)
        #
        #--- Get the fixed and free nodes and the displacement imposed
        #
        DOFfree, DOFfixed = self.getDOF(BC,nNodes)
        u_d = list(np.array(BC)[:,1])
        #
        #--- Applying the Boundary Conditions
        #
        K11 = K[DOFfree,:][:,DOFfree]
        K12 = K[DOFfree,:][:,DOFfixed]
        F = np.array((self.f_ext-self.f_int))
        F1 = F[DOFfree]
        #
        #--- Solution of the system : Ku = F
        #
        u_f = la.solve(K11, F1 - np.matmul(K12,u_d)) + self.U[DOFfree]
        self.U[DOFfixed] = u_d
        self.U[DOFfree] = u_f
        #
        #--- Integration of the behavior law and update plasticity parameters
        #
        self.eps = self.ComputeStrain(self.U)  
        model_pl = self.ComputePlastictyModel(x_nodes, Prop, struc)
        self.UpdatePlasticity(model_pl, x_nodes)
        #
        #--- Residus computation
        #
        self.f_int = S*V1.Linear_form(x_nodes,1,[],self.sigma)
        Res = self.f_ext - self.f_int
        Res_norm = la.norm(Res[DOFfree])
        #
        #--- Start of the interaction for the plasticity behavior
        #
        k = 0
        print('*** Iteration %02d: residual norm =%15.8e ***' % (k,Res_norm))
        
        while (Res_norm > tol) and k<itermax:
            
            K = S*M1.Matrix1D(x_nodes,1,1,[],self.H)
            K11 = K[DOFfree,:][:,DOFfree]
            du = la.solve(K11, Res[DOFfree])
            self.U[DOFfree] += du
            self.eps = self.ComputeStrain(self.U)
            model_pl =self.ComputePlastictyModel(x_nodes, Prop, struc)
            self.UpdatePlasticity(model_pl, x_nodes)
            self.f_int = S*V1.Linear_form(x_nodes, 1, [], self.sigma)
            Res = self.f_ext - self.f_int
            Res_norm = la.norm(Res[DOFfree])
            print('*** Iteration %02d: residual norm = %15.8e ***' % (k+1,Res_norm))
            k= k+1
       
        if k >= itermax:
            print('*** CONVERGENCE ERROR : Too many iteraction ***')
        elif Res_norm <= tol and k > 0:
            print('*** CONVERGED : residual norm = %.8e < tolerance = %.2e ***' % (Res_norm, tol))
        elif k == 0:
            print('*** CONVERGED : No plastic behavior / Structure remains in the elastic domain ***')

    def getDOF(self,BC,n):
        DOFfixed = list(np.array(BC)[:,0])
        DOFfree = list(set([i for i in range(n)]) - set(DOFfixed))
        return DOFfree, DOFfixed

    def ComputeStrain(self,U):
        eps = (U[1:] - U[:-1])/self.L_elt
        return np.array(eps)
   
    def ComputePlastictyModel(self, x_nodes, Prop, struc):
        model_pl = []
        deps = self.eps - struc['eps']
        for i in range(len(x_nodes)-1):
            hist = dict()
            hist['Eps_pl'] = struc['Eps_pl'][i]
            hist['sigma'] = struc['sigma'][i]
            hist['p'] = struc['p'][i]
            hist['H'] = struc['H'][i]
            hist['R'] = struc['R'][i]
            model_pl.append(BL.BehaviorLaw(Prop, hist, deps[i]))
        return model_pl
    
    def UpdatePlasticity(self,model_pl,x_nodes):
        for i in range(len(x_nodes)-1):
            self.Eps_pl[i] = model_pl[i].Eps_pl 
            self.sigma[i] = model_pl[i].Sigma 
            self.p[i] = model_pl[i].p 
            self.H[i] = model_pl[i].H 
            self.R[i] = model_pl[i].R
        return self