### Alex Weinberg
### NeoclassicalGrowthVFI
## Summer 2018

# Import packages
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.optimize import fminbound
from scipy import interpolate
from quantecon.markov.approximation import rouwenhorst

#### Parameters
gamma = 0.5     # risk aversion
beta = 0.95     # discount factor
delta = 0.05    # depreciation rate of capital
alpha = 0.4     # return to capital

# Computational Options
tol = 1e-8
maxiter = 100

'''
------------------------------------------------------------------------
Create Grid for State Space - Capital
------------------------------------------------------------------------
kmin      = scalar, lower bound of capital grid
kmax      = scalar, upper bound of capital grid
size_k    = integer, number of grid points in capital state space
k_grid    = vector, size_k x 1 vector of capital grid points
------------------------------------------------------------------------
'''
kmin   = 5 # Note that the steady state of k is 11.65 so we create grid around that
kmax   = 15
size_k = 10  # Number of grid points for k
k_grid = np.linspace(kmin, kmax, size_k)

'''
------------------------------------------------------------------------
Create Grid for State Space - Shocks to Production
------------------------------------------------------------------------
Random shocks to production are drawn
ln(z) ~ N[0,sigZ]

size_z    = scalar, number of grid points
sigZ      = scalar, variance of the shocks
muZ       = scalar, mean of the shocks
z_grid    = vector, state space of the shocks
dist      = markov chain object, see quantecon for details
pi        = matrix of markov chain (size_z x size_z)
rouwenhorst = method from quantecon of discretizing a markov chain
------------------------------------------------------------------------
'''
size_z = 5
sigZ = 0.2
muZ = 0
dist = rouwenhorst(size_z, muZ, sigZ, rho=0)
z_grid = np.exp(dist.state_values)
pi = np.transpose(dist.P)

def production(k,z=1):
    ''' Production function of capital and a random shock'''
    y = z * (k ** alpha) # makes things positive
    return y

def capital_transition(k_old,investment):
    ''' Transition equation of capital, takes in old capital and investment '''
    k_new = ((1 - delta) * k_old) + investment
    return k_new


@np.vectorize
def utility(c):
    c = np.maximum(c,1e-9)
    if gamma == 1:
        return np.log(c)
    else:
        u = (c ** (1 - gamma)) / (1 - gamma)
        return u

'''
------------------------------------------------------------------------------
Create grid of current utility values
------------------------------------------------------------------------------
C        = matrix, current consumption (c=z_tk_t^alpha - k_t+1)
U        = matrix, current period utility value for all possible
           choices of w and w' (rows are w, columns w')
------------------------------------------------------------------------------
'''

def expected_value(V,k,iz,j,sav):
    '''
    Takes in value function and current state and spits out
    expected_value for each savings decision
    '''
    EV = 0
    for ii, z_prime in enumerate(z_grid):
        k_tomorrow = capital_transition(k,sav)
        k_tomo_idx = np.argmin(np.abs(k_grid - k_tomorrow))
        EV += pi[iz, ii] * V[k_tomo_idx, ii]

    return EV
'''
------------------------------------------------------------------------
Value Function Iteration
------------------------------------------------------------------------
V         = vector, the value functions at each iteration
Vmat      = matrix, the value for each possible combination of k and k'
vvv       = vector, the updated value function
sav_idx   = vector, indicies of choices of w' for all w
VF        = vector, the "true" value function
------------------------------------------------------------------------
'''
Vguess = np.zeros((size_k)) # initial guess at value function
Vmat = np.zeros((size_k, size_k))


V = Vguess
diff = 7.0
its = 1
while diff > tol and its < maxiter:
    if its == 3:
        break

    for ik, k in enumerate(k_grid): # loop over state
        #for iz, z in enumerate(z_grid):
            y = production(k)
            for j,sav in enumerate(k_grid): # loop over choice
                con = y - sav
                Vmat[ik, j] = utility(con) + beta * V[j]

    vvv = Vmat.max(1)
    idx_sav = np.argmax(Vmat, axis=1)

    diff = (np.absolute(V - vvv)).max()  # check distance
    print("Iteration: ", its, ", Distance =  ", diff)
    its += 1

    V = vvv


VF = V # solution to the functional equation

if its < maxiter:
    print('Value function converged after this many iterations:', its)
else:
    print('Value function did not converge')

# Plot optimal consumption rule as a function of capital size# Plot o
plt.figure()
fig, ax = plt.subplots()
ax.plot(k_grid[:], VF, label='Consumption')
# Now add the legend with some customizations.
#legend = ax.legend(loc='upper left', shadow=False)
# Set the fontsize
#for label in legend.get_texts():
#    label.set_fontsize('large')
#for label in legend.get_lines():
#    label.set_linewidth(1.5)  # the legend line width
plt.xlabel('Size of Capital')
plt.ylabel('Optimal Consumption')
plt.title('Optimal consumption as size of capital')
plt.show()
