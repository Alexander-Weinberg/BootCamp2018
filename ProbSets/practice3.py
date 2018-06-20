## Root minimization in python 

### Section 1

# Root finders and minimizers are key to solving numerical optimization. Generally that problem is 

# argmin x f(x,z|θ)
# s.t. g(x,z|θ)≥0 and h(x)=0

# where f is a system of nonlinear equations (functions of x,z)
# x,z are vectors of variables and parameter vector θθ
# subject to the vector of inequality constraints gg and the vector of equality constraints hh.

# A computational algorithm that searches for the value of x that minimizes the problem above is called a minimizer. 
# Sometimes the solution to the minimization problem above can be written as a system of equations in xx.

# xhat = all x  s.t ϕ(x|z,θ)=0
# The maximization problem can be reduced to this system of characterizing equations when 
# the inequality constraints can be shown to never bind (interior solution). 
# A computational algorithm that searches for the value of xhat 
# that sets the value of each equation in the system ϕ(x|z) to 0 is called a root finder.

# ### Section 2, root finders
# suppose solution to a system of N nonlinear equations ϕ(x|z,θ)=0 is Nx1 vector xhat

def phi_pol(xvals, theta):
    '''
    --------------------------------------------------------------------
    This function returns the value phi(x,theta) given x and theta,
    where phi(x,theta) = (x ** 3) + x + theta
    --------------------------------------------------------------------
    INPUTS:
    xvals = scalar or (N,) vector, value or values for x
    args  = length 1 tuple, (theta)
    
    OTHER FUNCTIONS AND FILES CALLED BY THIS FUNCTION: None
    
    OBJECTS CREATED WITHIN FUNCTION:
    theta = scalar, constant in the phi function
    phi   = scalar or (N,) vector, value of phi(xvals, theta)
    
    FILES CREATED BY THIS FUNCTION: None
    
    RETURNS: phi
    --------------------------------------------------------------------
    '''
    phi = (xvals ** 3) + xvals + theta
    
# Can use this to explore grahically
# answer should be 2

# Scipy has scipy.optimize, which has most univariate, multivariate root finding algos
# as well as Python's most standard minimizer algorithms. 

# brentq()
# brenth()
# ridder()
# bisect()
# newton() # does not require bracketing interval for root, rest do

import scipy.optimize as opt

theta = 10
# search grid 
a =-5
b = 0

# (xhat, result) = opt.bisect(phi_pol, a, b, args=(theta,), full_output=True)
x_init = -10.0

xhat = opt.newton(phi_pol, x_init, args=(theta,))
print(xhat)
