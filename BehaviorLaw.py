'''
This module allows the calcule of the elastoplastic behavior law
considering a isotrope hardening in power law: R(p) = B*p^n
'''

import function
import numpy as np

class BehaviorLaw:
    def __init__(self,prop,hist,deps, nGauss = 1):
        '''
        prop: dictionary with material properties
        nGauss: number of Gauss point        
        deps: increment in strain in each step       
        '''
        self.E = prop['E']
        self.B = prop['B']
        self.Sigma_y = prop['Sigma_y']
        self.n = prop['n']
        self.deps = deps
        
        self.Eps_pl = hist['Eps_pl']
        self.Sigma = hist['sigma']
        self.p = hist['p']
        self.R = hist['R']
        self.H = hist['H']
        
        for G in range(nGauss):
            self.sigma_star = self.Sigma+self.E*self.deps
            f_star = abs(self.sigma_star) - self.R - self.Sigma_y
            if f_star <= 0:
                self.Sigma = self.sigma_star
                self.Eps_pl = self.Eps_pl
                self.p = self.p
                self.R = self.R
                self.H = self.E
            else:
                delta_p = function.newton(self.Computef,1e-14,1e-3)
                self.Sigma = self.sigma_star-self.E*delta_p*np.sign(self.sigma_star)
                self.Eps_pl = self.Eps_pl+delta_p*np.sign(self.sigma_star)
                self.p = self.p+delta_p
                self.R = self.ComputeR(self.p)
                self.H = self.ComputeH(self.p)
        
    def ComputeH(self,p):
        return self.E *(1 - np.sign(self.Sigma)*self.E*np.sign(self.sigma_star)/(self.E*np.sign(self.sigma_star) + self.B*self.n*p**(self.n-1)))
    
    def ComputeR(self,p):
        return self.B*(p)**self.n
    
    def ComputedR(self,delta_p):
        return self.n*self.B*(delta_p)**(self.n-1)
    
    def Computef(self,delta_p):
        f = abs(self.sigma_star) - self.E*delta_p - self.ComputeR(self.p+delta_p) - self.Sigma_y
        df = -self.E-self.ComputedR(self.p+delta_p)
        return (f,df)
    