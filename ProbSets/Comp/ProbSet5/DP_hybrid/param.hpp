/* ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 
 * param.hpp
 * 
 * This file declares the parameters of the problem. It 
 * declares the preference parameters, the state space, the 
 * level of discretization, and the number of iterations. 
 * It also declares parameters useful for 
 * formatting the output files of the program.
 * 
 * Carlos Rangel & Simon Scheidegger -- 06/17.
 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/ 

#ifndef param_hpp
#define param_hpp

#ifndef EIGEN_DONT_PARALLELIZE
#define EIGEN_DONT_PARALLELIZE 1 // Disables Eigen's OpenMP application to matrix operations
#endif

#include <stdio.h>
#include "../eigen/Eigen/Dense"

using namespace Eigen;

// Preference Parameters
extern double alpha;
extern double gamm;
extern double beta;

// Capital Stock Interval

// Minimum and Maximum Capital Stock
extern double kmin;
extern double kmax;

// Number of capital stocks and discretization level
extern int nk;
extern double kappa;


// Number of theta states and the minimum and maximum theta values
extern int ntheta;
extern double theta_min;
extern double theta_step;

// Capital and Theta grids
extern ArrayXd kgrid;
extern ArrayXd thetagrid;

// Number of Iterations
extern int numstart; // Iterations begin at this timestep
extern int Numits;   // iterations end at this timestep

// Maximum difference between succesive Value Function Iterates (initialized to 0)
extern double errmax;

// Number of Columns for output
extern int nout;

// Frequency with which data will be printed
extern int datafreq;

#endif /* param_hpp */
