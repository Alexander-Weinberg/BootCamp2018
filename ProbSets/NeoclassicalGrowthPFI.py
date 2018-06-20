#### Alex Weinberg 
### NeoclassicalGrowthPFI
## Summer 2018

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.optimize import fminbound
from scipy import interpolate

#### Tolerance options
tol = 1e-5
maxiter = 100

#### Parameters 
gamma = 0.5
beta = 0.95
delta = 0.05
alpha = 0.4

##### Asset space
kmin = 0
kmax = 100
nk = 100

kgrid = np.linspace(kmin,kmax,nk)

#### Discretized risk
sigZ = 0.2 
muZ = 0
# for now, z_t =1 , update later
zt=1

#===============================
# define utility and marginal utility funcs
def utility(c):
    ''' CRRA Utility with gamma = risk aversion '''
    if gamma == 1:
        return np.log(c)
    else:
        return c ** (1 - gamma) / (1 - gamma)

def u_prime(c):
    return c ** -gamma

#===============================
def coleman_operator(phi, w_grid):
    '''
    The Coleman operator, which takes an existing guess phi of the
    optimal consumption policy and computes and returns the updated function
    Kphi on the grid points.
    '''

    # === Apply linear interpolation to phi === #
    phi_func = interpolate.interp1d(w_grid, phi, fill_value='extrapolate')

    # == Initialize Kphi if necessary == #
    Kphi = np.empty_like(phi)

    # == solve for updated consumption value
    for i, w in enumerate(w_grid):
        def h(c):
            return u_prime(c) - beta * u_prime(phi_func(R * (w - c)))
        results = opt.root(h, 1e-10)
        c_star = results.x[0]
        Kphi[i] = c_star
        
    return Kphi

