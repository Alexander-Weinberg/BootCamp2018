#### Alex Weinberg
### NeoclassicalGrowthPFI
## Summer 2018

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.optimize import fminbound
from scipy import interpolate

#### Parameters
gamma = 1
beta = 0.95
delta = 0.05
alpha = 0.4

##### Asset space
kss = ((1 - delta - (1/beta)) / alpha) ** (1/(alpha-1)) # analytically solved for steady state

nk = 200
kmin = 0
kmax = kss + 2
nk = 200

kgrid = np.linspace(kmin,kmax,nk)

#### Discretized risk
sigZ = 0.2
muZ = 0
# for now, z_t =1 , update later
zt=1

#===============================
# define Functions of problem 
def utility(c):
    ''' CRRA Utility with gamma = risk aversion '''
    if c <= 0:
        c = 1e-8
    if gamma == 1:
        return np.log(c)
    else:
        return (c ** (1 - gamma)) / (1 - gamma)

def u_prime(c):
    if c <= 0:
        c = 1e-8  
    return 1/ (c ** gamma)

def production(k):
    y = zt * (k ** alpha)
    return y

def capital_transition(k,i):
    # k = capital
    # i = savings
    kap_tomorrow = (1 - delta) * k + i
    return kap_tomorrow
  
#===============================

def coleman_operator(phi, kgrid):
    '''
    The Coleman operator, which takes an existing guess phi of the
    optimal consumption policy and computes and returns the updated function
    Kphi on the grid points.
    '''

    # === Apply linear interpolation to phi === #
    phi_func = interpolate.interp1d(kgrid, phi, fill_value='extrapolate')

    # == Initialize Kphi if necessary == #
    Kphi = np.empty_like(phi)

    # == solve for updated consumption value
    for ik, k in enumerate(kgrid):
        def h(c):
            y = production(k) # income from capital
            kap_tomorrow = capital_transition(k, y-c)
            return u_prime(c) - beta * u_prime(phi_func(kap_tomorrow))
        results = opt.root(h, 1e-10)
        c_star = results.x[0]
        Kphi[ik] = c_star

    return Kphi
#============================================
'''
------------------------------------------------------------------------
Policy Function Iteration
------------------------------------------------------------------------
tol     = scalar, tolerance required for policy function to converge
diff   = scalar, distance between last two policy functions
maxiter = integer, maximum number of iterations for policy function
phi       = vector, policy function for choice of consumption at each iteration
iter    = integer, current iteration number
new_phi   = vector, updated policy function after applying Coleman operator
------------------------------------------------------------------------
'''
#### Tolerance options
tol = 1e-5
maxiter = 100
diff = 7

conguess = kgrid ** alpha  # consume income
con = conguess

iter = 1
while diff > tol and iter < maxiter:
    new_con = coleman_operator(con, kgrid)
    diff = (np.absolute(con - new_con)).max()
    con = new_con
    print('Iteration ', iter, ' distance = ', diff)
    iter += 1

pol_con = con
pol_sav = kgrid - con

#####################################################
# PLOTTING
#####################################################
plt.figure()
fig, ax = plt.subplots()
ax.plot(kgrid[1:], pol_con[1:], label="Consumption policy")
# Now add the legend with some customizations.
legend = ax.legend(loc='upper left', shadow=False)
# Set the fontsize
for label in legend.get_texts():
    label.set_fontsize('large')
for label in legend.get_lines():
    label.set_linewidth(1.5)  # the legend line width
plt.xlabel('Kgrid')
plt.ylabel('Optimal Consumption')
plt.title('Policy Function, consumption - NCG')
plt.show()
