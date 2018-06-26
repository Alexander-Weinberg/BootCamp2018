# Import packages
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.optimize import fminbound
from scipy import interpolate
from quantecon.markov.approximation import rouwenhorst

# Quesiton 2
## PARAMETERS
gamma = 0.5
beta = 0.96
delta = 0.05
alpha = 0.4
sigz = 0.2
muz = 0

# Discretize captial
kmin = 10
kmax = 13
nk = 30

kgrid = np.linspace(kmin,kmax,nk)

# Discretize risk
nz = 30
zdist = rouwenhorst(nz, muz, sigz, rho=0)
zgrid = np.exp(zdist.state_values)
pi = zdist.P

# Options
tol = 1e-4
maxiter = 1000


'''
 --------------------------------------------------------------------------------
Create grid of current utility values
------------------------------------------------------------------------
C        = matrix, current consumption (c=z_tk_t^a - k_t+1 + (1-delta)k_t)
U        = matrix, current period utility value for all possible
           choices of w and w' (rows are w, columns w')
------------------------------------------------------------------------
'''

C = np.zeros((nk, nk, nz))
for i in range(nk): # loop over k_t
    for j in range(nk): # loop over k_t+1
        for q in range(nz): #loop over z_t
            C[i, j, q] = zgrid[q]* kgrid[i]**alpha + (1 - delta)*kgrid[i] - kgrid[j]
# replace 0 and negative consumption with a tiny value
# This is a way to impose non-negativity on cons
C[C<=0] = 1e-15
if gamma == 1:
    U = np.log(C)
else:
    U = (C ** (1 - gamma)) / (1 - gamma)
U[C<0] = -9999999


def production(k,z=1):
    y = z * (k ** alpha)
    return y

def capital_transition(k,sav):
    knew = (1 - delta) * k + sav
    return knew

def expected_value(Vlast,k,iz,sav):
    '''
    V = value func
    k = current capital
    iz = index of current shock
    Takes in value function and current state and spits out
    expected_value for each savings decision
    '''

    EV = 0
    for ii, z_prime in enumerate(zgrid):
        V_func = interpolate.interp1d(kgrid, Vlast[:,ii], kind='cubic', fill_value='extrapolate')

        k_tomo = capital_transition(k,sav)

        EV += pi[iz, ii] * V_func(k_tomo)
    return EV
################
#VFI
################
'''
------------------------------------------------------------------------
Value Function Iteration
------------------------------------------------------------------------
VFtol     = scalar, tolerance required for value function to converge
VFdist    = scalar, distance between last two value functions
VFmaxiter = integer, maximum number of iterations for value function
V         = vector, the value functions at each iteration
Vmat      = matrix, the value for each possible combination of w and w'
Vstore    = matrix, stores V at each iteration
VFiter    = integer, current iteration number
TV        = vector, the value function after applying the Bellman operator
PF        = vector, indicies of choices of w' for all w
VF        = vector, the "true" value function
------------------------------------------------------------------------
'''
VFtol = 1e-4
VFdist = 7.0
VFmaxiter = 500
V = np.zeros((nk, nz)) # initial guess at value function
Vmat = np.zeros((nk, nk, nz)) # initialize Vmat matrix
Vstore = np.zeros((nk, nz, VFmaxiter)) #initialize Vstore array
VFiter = 1
while VFdist > VFtol and VFiter < VFmaxiter:
    print('Iteration', VFiter, 'Distance,', VFdist)
    for i in range(nk): # loop over k_t
        for j in range(nk): # loop over k_t+1
            for q in range(nz): #loop over z_t
                EV = 0
                for qq in range(nz):
                    EV += pi[q, qq]*V[j, qq]
                Vmat[i, j, q] = U[i, j, q] + beta * EV

    Vstore[:,:, VFiter] = V.reshape(nk, nz,) # store value function at each iteration for graphing later
    TV = Vmat.max(1) # apply max operator over k_t+1
    PF = np.argmax(Vmat, axis=1)
    VFdist = (np.absolute(V - TV)).max()  # check distance
    V = TV
    VFiter += 1



if VFiter < VFmaxiter:
    print('Value function converged after this many iterations:', VFiter)
else:
    print('Value function did not converge')


VF = V # solution to the functional equation

# Plot value function
plt.figure()
fig, ax = plt.subplots()
ax.plot(kgrid[1:], VF[1:, 0], label='$z$ = ' + str(kgrid[0]))
ax.plot(kgrid[1:], VF[1:, 5], label='$z$ = ' + str(kgrid[5]))
ax.plot(kgrid[1:], VF[1:, 15], label='$z$ = ' + str(kgrid[15]))
ax.plot(kgrid[1:], VF[1:, 19], label='$z$ = ' + str(kgrid[19]))
# Now add the legend with some customizations.
legend = ax.legend(loc='lower right', shadow=False)
# Set the fontsize
for label in legend.get_texts():
    label.set_fontsize('large')
for label in legend.get_lines():
    label.set_linewidth(1.5)  # the legend line width
plt.xlabel('Size of Capital')
plt.ylabel('Value Function')
plt.title('Value Function')
plt.savfig('1.png')
plt.show()

#Plot optimal consumption rule as a function of capital
optK = kgrid[PF]
optC = kgrid * kgrid ** (alpha) + (1 - delta) * kgrid - optK
plt.figure()
fig, ax = plt.subplots()
ax.plot(kgrid[:], optC[:][18], label='Consumption')
# Now add the legend with some customizations.
#legend = ax.legend(loc='upper left', shadow=False)
# Set the fontsize
for label in legend.get_texts():
    label.set_fontsize('large')
for label in legend.get_lines():
    label.set_linewidth(1.5)  # the legend line width
plt.xlabel('Size of Capital')
plt.ylabel('Optimal Consumption')
plt.title('Policy Function, consumption - growth model')
plt.savfig('2.png')
plt.show()

#Plot optimal capital in period t + 1 rule as a function of cake size
optK = kgrid[PF]
plt.figure()
fig, ax = plt.subplots()
ax.plot(kgrid[:], optK[:][18], label='Capital in period t+1')
# Now add the legend with some customizations.
#legend = ax.legend(loc='upper left', shadow=False)
# Set the fontsize
for label in legend.get_texts():
    label.set_fontsize('large')
for label in legend.get_lines():
    label.set_linewidth(1.5)  # the legend line width
plt.xlabel('Size of Capital in period t')
plt.ylabel('Optimal Capital in period t+1')
plt.title('Policy Function, capital next period - growth model')
plt.savfig('3.png')
plt.show()
