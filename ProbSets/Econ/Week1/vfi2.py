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
tol = 1e-4
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
    y = z * (k ** alpha)
    return y

def capital_transition(k_old,investment):
    ''' Transition equation of capital, takes in old capital and investment '''
    k_new = ((1 - delta) * k_old) + investment
    return k_new

'''
------------------------------------------------------------------------------
Create grid of current utility values
------------------------------------------------------------------------------
C        = matrix, current consumption (c=z_tk_t^alpha - k_t+1)
U        = matrix, current period utility value for all possible
           choices of w and w' (rows are w, columns w')
------------------------------------------------------------------------------
'''

C = np.zeros((size_k, size_k,size_z))

for i, k in enumerate(k_grid): # loop over k
    for j, sav in enumerate(k_grid): # loop over k'
        for iz, z in enumerate(z_grid):
            C[i, j, iz] = production(k,z) - sav

# Impose non-negativity on consumption
C[C<=0] = 1e-8
if gamma == 1:
    U = np.log(C)
else:
    U = (C ** (1 - gamma)) / (1 - gamma)
U[C<0] = -np.inf

'''
------------------------------------------------------------------------
Value Function Iteration
------------------------------------------------------------------------
V         = vector, the value functions at each iteration
Vmat      = matrix, the value for each possible combination of k and k'
vvv       = vector, the updated value function after applying the Bellman operator
PF        = vector, indicies of choices of w' for all w
VF        = vector, the "true" value function
------------------------------------------------------------------------
'''
Vguess = np.zeros((size_k, size_z)) # initial guess at value function
Vmat = np.zeros((size_k, size_k, size_z))

V = Vguess
diff = 7.0
its = 1
while diff > tol and its < maxiter:
    its += 1
    for ik, k in enumerate(k_grid): # loop over state
        for iz, z in enumerate(z_grid):
            for j, sav in enumerate(k_grid): # loop over savings decision
                EV = 0
                for ii, z_prime in enumerate(z_grid):
                    k_tomorrow = capital_transition(k,sav)
                    k_tomo_idx = np.argmin(np.abs(k_grid - k_tomorrow))
                    EV += pi[iz, ii] * V[k_tomo_idx, ii]

            Vmat[ik, j, iz] = U[ik, j, iz] + beta * EV
    vvv = Vmat.max(1)
    sav_ind = np.argmax(Vmat, axis=1)
    diff = (np.absolute(V - vvv)).max()  # check distance
    print("Iteration: ", its, ", Distance =  ", diff)
    V = vvv


if its < maxiter:
    print('Value function converged after this many iterations:', its)
else:
    print('Value function did not converge')


VF = V # solution to the functional equation


middle = 2
# Plot optimal consumption rule as a function of capital size# Plot o
optK = k_grid[sav_ind]
# optC = z_grid * k_grid ** (alpha) + (1 - delta) * k_grid - optK
plt.figure()
fig, ax = plt.subplots()
ax.plot(k_grid[:], optK[:], label='Consumption')
# Now add the legend with some customizations.
#legend = ax.legend(loc='upper left', shadow=False)
# Set the fontsize
for label in legend.get_texts():
    label.set_fontsize('large')
for label in legend.get_lines():
    label.set_linewidth(1.5)  # the legend line width
plt.xlabel('Size of Capital')
plt.ylabel('Optimal Consumption')
plt.title('Optimal consumption as size of capital')
plt.show()
#def bellman_operator(V,EV):
#    '''
#    The approximate Bellman operator, which computes and returns the
#    updated value function TV on the grid points.  An array to store
#    the new set of values TV is optionally supplied (to avoid having to
#    allocate new arrays at each iteration).  If supplied, any existing data in
#    Tw will be overwritten.
#    '''
#
#    # Apply cubic interpolation to V
#    V_func = interpolate.interp1d(kgrid, V, kind='cubic', fill_value='extrapolate')
#
#    # Initialize array for operator and policy function
#    newV = np.empty_like(V)
#    optSav = np.empty_like(newV)
#
#    for ik, k in enumerate(kgrid):
#        def objective(sav):
#            return - utility(k-sav) - beta * V_func(sav)
#        root = fminbound(objective, -1e-5, k - 1e-6)
#        optSav[i] = root
#        newV[i] = - objective(root)
#    return newV, optSav
##===============================
#
#Vguess = np.zeros((size_eps, ksize)) # initial guess at value function
#V = Vguess
#
#Val = np.zeros((size_eps, ksize))
#Sav = np.zeros((size_eps, ksize))
#
#its = 1
#diff = 7.0
#while diff > tol and its < maxiter:
#    for i in range(size_eps): # loop over states
#        for j,k in enumerate(kgrid): #
#            for idx, sav in enumerate(kgrid):
#                vvv[i,j,idx] =
#            for ii in range(size_eps):  # loop over epsilon' to compute expected value
#                EV += pi[i, ii] * V[ii, j]
#
#            vvv = utility(k - kgrid) + beta * EV[]
#
#
#
#    diff = (np.absolute(V - TV)).max()  # check distance
#    V = TV
#    its += 1
#
#if VFiter < VFmaxiter:
#    print('Value function converged after this many iterations:', VFiter)
#else:
#    print('Value function did not converge')
#
#
#VF = V # solution to the functional equation
