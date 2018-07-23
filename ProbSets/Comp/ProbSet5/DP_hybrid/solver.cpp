/* ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 
 * solver.cpp
 * 
 * This defines the ValIt function, which takes the Old Value 
 * Function Guess and performs one Value function iteration.
 * 
 * Since we are using MPI to parallelize over the capital states, 
 * the MPI_COMM_WORLD communicator splits the nk states among the MPI processes within it and collects  
 * the results in the rank 0 process of that communicator. Furtheremore, since we are also using OpenMP, 
 * each MPI process creates threads and distributes the capital states that were assigned to it among those threads.
 * 
 * Carlos Rangel & Simon Scheidegger -- 06/17.
 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/ 

#include <iostream>
#include <stdio.h>
#include "param.hpp"
#include "econ.hpp"
#include "omp.h"
#include <mpi.h>

using namespace Eigen;


// Value Function Iteration

// The following defines the routine for one iteration of value function iteration, and produces the new
// value function. This is the Bellman equation

// In this problem, the control (knext here) is discrete, so maximization reduces to computing a list of
// values implied by all possible choices, and taking the max.

MatrixXd ValIt(MatrixXd ValOld=MatrixXd::Zero(nk, ntheta)) {
    
    // Create Index Variable for Storing the location (index) of the optimal policy choice
    MatrixXd::Index maxIndex;
    
    
    // The following array will contain the consumption choices implied by each policy
    ArrayXd c(nk);
    
    // Temp will contain the list of values implied by each policy choice
    ArrayXd temp(nk);

    // Processes' ID's inside MPI_COMM_WORLD and the size of said Communicator

    int my_rank, size;
    
    // Obtain number of proccesses and id's
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    /* We now create and initialize ValNew, which will store the new value 
     * function guess after the iteration. We initially set ValNew equal 
     * to the Old Value Function Guess. Only world rank 0 needs to store 
     * the New Value Function. Thus, all other processes define ValNew 
     * to be a 1x1 matrix of zeros in order to save space in memory.
     */
    
    MatrixXd ValNew= (my_rank==0) ? ValOld: MatrixXd::Zero(1,1);
    
    /* Declare and Initialize the Policy Function to a Matrix of Zeros. Only the process with rank 0 inside
     MPI_Comm_World needs to store the complete (nk x ntheta) matrix. Thus, all other processes define
     Policy to be a 1x1 matrix of zeros in order to save space in memory.
     */
    
    MatrixXd Policy=(my_rank==0) ? MatrixXd::Zero(nk, ntheta): MatrixXd::Zero(1,1);
    
    /* We now proceed to split the number of capital states among the MPI Processes within MPI_Comm_World.
     Each process will compute the new values and optimal policies corresponding to those capital states.
     
     We try to divide those capital states among the processes as evenly as posisble. If the number of
     capital states is not divisible by the number of mpi processes, the first r processes get
     an extra capital stock each, where r is the remainder after dividing nk by size.
     
     */

    // Determine approximately how many capital states are assigned to each process.
    int nump=nk/size;
    
    // Obtain the remainder from the previous division
    int r=nk % size;
    
    
    /*
     
     Assuming that we have a zero remainder, the idea is to have process 0 work on the first
     nump capital states, process 1 on the next nump capital states, and so on. If we have a
     nonzero remainder r, we will give each of the first r processes an extra capital state.
     
     The following variable indicates where each process' kgrid segment starts.
     
     */
    
    
    int my_start;
    
    /* The following determines how many capital states each mpi process gets. It also
     determines the value for my_start.
     */
    
    
    if (my_rank<r) {
        nump+=1;
        my_start=my_rank*nump;
    }
    
    // We now determine the starting indices of the processes that were not assigned
    // a remainder.
    
    else {
        my_start=r + my_rank*nump;
    }
    
    /* Create the kgrid containing the k's for which the process will compute the
     new values and policies
     */
    ArrayXd my_kgrid=kgrid.segment(my_start, nump); // Segment of nump capital stocks starting at kgrid(my_start)
    
    // Each process will store the new values and policies corresponding to their states in the
    // following buffers. These will be then sent to the world rank 0 process.
    ArrayXd val_sendbuff=ArrayXd::Zero(nump);
    ArrayXd pol_sendbuff=ArrayXd::Zero(nump);
    
    /* Receive Buffers. Process with world rank 0 will gather the results sent by the other processes in the
     following buffers, which will then be assigned to the New Value function and Policy matrices.
     Only world rank 0 process needs to store the complete nk array */
    
    
    ArrayXd val_recvbuff= (my_rank==0) ? ArrayXd::Zero(nk) : ArrayXd::Zero(1);
    ArrayXd pol_recvbuff= (my_rank==0) ? ArrayXd::Zero(nk) : ArrayXd::Zero(1);
    
    // Create "Displacement Array" for MPI_Gatherv: Indicates where each process will start saving their
    // respective data in the receive buffer.
    
    int displs[size];
    
    
    // Create array that indicates how many elements we are going to receive from each process
    
    int recvcounts[size];
    
    
    /* The first r processes will each send 1+ nk/size elements to each of 
     * the receive buffers, since they each were assigned an extra capital stock. 
     * Thus, their starting points must be spaced out accordingly.
     */
    
    for (int i=0; i<r; i++) {
        displs[i]=i*(1+nk/size);
        recvcounts[i]=1+nk/size;
    }
    
    // The remaining procceses only store nk/size elements. Their starting points are spaced out
    // accordingly as well.
    for (int j=r; j<size; j++) {
        displs[j]=r + j*nk/size;
        recvcounts[j]=nk/size;
    }
    
    
    /*
     The following is the economics part of the program. For each state (theta, k), 
     we compute the new value and policy functions. Since the problem is discrete, 
     the problem reduces to computing the values associated with each policy choice
     and choosing the one that yields the highest value. 
     These policies and new values will then be part of our new value and policy function guesses.
     
     The following makes use of MPI and OpenMP to parallelize over 
     the capital states. As mentioned before, we distribute the capital 
     states among the MPI processes, each of which then distributes those that 
     were assigned to it among the threads it creates
     */

    
    for (int itheta=0; itheta<ntheta; itheta++) {
        
        // OpenMP is initialized and the threads are created within each MPI process.
#pragma omp parallel private(maxIndex, c, temp)
        {
            // We distribute the for loop among the process' threads
            /*
             
             Given the theta state, we now determine the new values and optimal policies corresponding to each
             capital state.
             
             */
#pragma omp for
            for (int ik=0; ik<nump; ik++) {
                
                // Compute the consumption quantities implied by each policy choice
                c=f(my_kgrid(ik), thetagrid(itheta))-kgrid;
                
                // Compute the list of values implied implied by each policy choice
                temp=util(c) + beta*ValOld*p(thetagrid(itheta));
                
                /* Take the max of temp and store its location.
                 The max is the new value corresponding to (ik, itheta).
                 The location corresponds to the index of the optimal policy choice in kgrid.
                 
                 We store the new value and the corresponding policy information in the send buffers declared earlier
                 
                 */
                
                val_sendbuff(ik)=temp.maxCoeff(&maxIndex);
                pol_sendbuff(ik)=kgrid(maxIndex);
            }
   
        }
    
        
        
        /* Given the theta state, World Rank 0 gathers the new values and policies computed by all processes and
         stores them in the receive buffers declared earlier.
         */

        MPI_Gatherv(val_sendbuff.data(), nump, MPI_DOUBLE, val_recvbuff.data(), recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        MPI_Gatherv(pol_sendbuff.data(), nump, MPI_DOUBLE, pol_recvbuff.data(), recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        
        
        /* Now that we have obtained the new values and policies for all nk states for a given theta, we update
         the Value and policy matrices in the world rank 0 process. That is, we assign the receive buffers to the
         corresponding columns in the Value and Policy function matrices.
         */

        if (my_rank==0) {
            ValNew.col(itheta)=val_recvbuff;
            Policy.col(itheta)=pol_recvbuff;
        }
        
    }
    
    
    
    // World Rank 0 Process concatenates ValNew and Policy into a single (2nk x theta_ntheta) matrix and returns it
    
    if (my_rank==0) {
        MatrixXd result(2*nk, ntheta);
        
        result<< ValNew, Policy;
        
        return result;
    }
    
    // All other processes just return ValNew, which is just a 1x1 Zero matrix outside of world rank 0.
    
    else {
        return ValNew;
    }
}
