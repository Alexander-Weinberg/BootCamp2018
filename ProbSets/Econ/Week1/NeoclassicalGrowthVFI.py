#### Alex Weinberg
### NeoclassicalGrowthVFI
## Summer 2018

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.optimize import fminbound
from scipy import interpolate
from quantecon.markov.approximation import tauchen

#### Parameters
gamma = 0.5       # risk aversion
beta = 0.95     # discount factor
delta = 0.05    # depreciation rate of capital
alpha = 0.4     # return to capital

# Computational Options
tol = 1e-5
maxiter = 1000

### Discretize Capital State space
kmin = 0
kmax = 10
ksize = 100

kgrid = np.linspace(kmin,kmax,ksize)

### Utility
@np.vectorize
def utility(c):
    ''' CRRA Utility function '''
    if c <= 0:
        c = 1e-5
    if gamma == 1:
        return np.log(c)
    else:
        return (c ** (1 - gamma)) / (1 - gamma)

@np.vectorize
def production(k,z=1):
    ''' Production function of capital and a random shock'''
    y = z * (k ** alpha)
    return y

@np.vectorize
def capital_transition(k_old,investment):
    ''' Transition equation of capital, takes in old capital and investment '''
    k_new = ((1 - delta) * k_old) + investment
    return k_new

# write Bellman operator function to help with VFI
def bellman_operator(Vlast):
    '''
    Bellman operator. 1) Takes in old estimate of value function. Interpolates
    the old value function to allow more flexibility. 2) for each element of k state,
    it find the root (max savings decision)
    3) returns array of optimal savings decisions for each capital input
    4) returns a value function for each of those capital input

    Arguments:
    Vlast(array): take in old estimate of value function

    Returns:
    newV(array): new estimate of value function
    optSav(array): new estimate of value function

    '''
    V_func = interpolate.interp1d(kgrid, Vlast, kind='cubic', fill_value='extrapolate')

    # Initialize array for operator and policy function
    newV = np.empty_like(Vlast) # value func
    polSav = np.empty_like(newV) # savings policy
    vvv = np.zeros((ksize,ksize))

    # == set TV[i] = max_sav { u(con) + beta V(sav)} == #
    for iK, k in enumerate(kgrid):
        for iS, sav in enumerate(kgrid):
            y = production(k) # output
            k_new = capital_transition(k,sav) # new capital for each saving decision
            # Value today, we want to maximize this
            vvv[iK,iS] = utility(y - sav) - beta * V_func(k_new)

    ind_sav = np.argmax(vvv, axis=1)
    polSav = kgrid[ind_sav]
    newV = np.amax(vvv, axis=1)


    return newV, polSav

Vguess = utility(kgrid)
V = Vguess

diff = 7.0
its = 0
while diff > tol and its < maxiter:
    Vlast = V

    V, polSav = bellman_operator(Vlast)
    diff = (np.absolute(V - Vlast)).max()

    print('Iteration ', its, ', distance = ', diff)
    its += 1

polCon = kgrid - polSav

# Plot value function
plt.figure()
plt.plot(kgrid[1:], polSav[1:])
plt.xlabel('Input Capital')
plt.ylabel('Savings Function')
plt.title('Savings Function - Neoclassical Growth Model')
plt.show()
