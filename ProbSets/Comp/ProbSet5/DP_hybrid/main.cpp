/* ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 
 * main.cpp
 * 
 * In this file, we perform and time Value Function Iterations. 
 * Furthermore, we store the results of certain 
 * iterations in text files. We end by computing the time it takes 
 * to write the results of a single iteration. 
 * 
 * Carlos Rangel & Simon Scheidegger -- 06/17.
 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/ 


#include <iostream>
#include <fstream>
#include "param.hpp"
#include "econ.hpp"
#include <mpi.h>
#include "omp.h"

using namespace Eigen;
using namespace std;

int main(int argc, char *argv[]) {
    
    // time variables
    double t1, t2;
    
    /* Declare the Value Function Iteration Function. This function takes the old value f
    unction guess as an argument and returns a 2nk x ntheta matrix consisting of a 
    concatenation of the new value function guess and the policy matrix. 
    More specifically, the first nk x ntheta block of this matrix is the 
    new value function guess whereas the last block is the Policy function.*/
    
    MatrixXd ValIt(MatrixXd ValOld);
    
    
    /* Declare Old Value Function Matrix. We will store our initial guess 
    for the value function here. Every process must have the Old Value 
    function at its disposal. Therefore, it is declared before initializing MPI.*/
    MatrixXd ValOld(nk, ntheta);
    
    
    // Unit for Printing Data to text files
    ofstream output;
    
    // Dummy Variable used for storing the return value of scanf (otherwise, we get compiler warnings)
    int dumm;
    
    // These variables will store the MPI process' ID with respect to the MPI_Comm_World Communicator
    // and the size of said communicator.
    int my_rank, size;
    
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    // Obtain number of proccesses and id's
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Declare and Initialize the Policy Function to a Matrix of Zeros. Only the process with rank 0 inside
    // MPI_Comm_World needs to store the complete (nk x ntheta) matrix. Thus, all other processes define
    // Policy to be a 1x1 matrix of zeros in order to save space in memory.

    MatrixXd Policy=(my_rank==0) ? MatrixXd::Zero(nk, ntheta): MatrixXd::Zero(1,1);
    
    /* Result is a (2nk x final_ntheta) matrix that will store the matrix returned by ValIt. 
    Only process with rank 0 inside MPI_Comm_World will need to store the full matrix. 
    Thus, Result will be a 1x1 matrix on all other processes.*/
    MatrixXd Result=(my_rank==0) ? MatrixXd::Zero(2*nk, ntheta) : MatrixXd::Zero(1,1);
    
    /* Matrix for storing Data in desired format. This matrix will be used by process
    with rank 0 inside MPI_Comm_World to store the results of the iterations in text files. 
    Since this process will be doing all of the printing, it is the only one storing the full matrix.*/
    
    MatrixXd Display= (my_rank==0) ? MatrixXd::Zero(nk, nout) : MatrixXd::Zero(1,1);
    
    
    /*
     ValOld and Policy are set equal to the Value and Policy functions stored during the numstart iteration.
     If numstart is equal to zero, they are simply set equal to zero matrices.
     
     We initialize the matrices inside the world rank 0 process, which will later boradcast the necessary information to the rest of the processes.
     */

    
    if (my_rank==0) {
        if (numstart==0) {
            ValOld=MatrixXd::Zero(nk, ntheta);
        }
        
        // Extract Value and Policy functions from the corresponding text files.
        else {
            float f,g;  // Variables that will store the values of the value and policy function, respectively.
            FILE * pFile;   // Pointer to the text file
            
            for (int j=0; j<ntheta; j++) {
                char filename_dummy[100];
                sprintf(filename_dummy,"%s%.2f%s%d%s", "results/theta_" ,thetagrid(j),"_", numstart,".txt");
                pFile=fopen(filename_dummy, "r+");
                
                dumm=fscanf(pFile, "%*s %*s %*s");  // Dump the initial row consisting of strings describing the columns of the data file.
                
                for (int i=0; i<nk; i++) {
                    
                    // Dump the value of the first column (corresponding to the kgrid) and extract the values of the value and policy functions in the text file.
                    dumm=fscanf (pFile, " %*f %f %f ", &f, &g);
                    ValOld(i,j)=f;
                    Policy(i,j)=g;
                }
                
                fclose(pFile);
            }
            
        }

        
    }
    
    /* We now create and initialize ValNew, which will store the new value function 
     * guess after each iteration. We initially set ValNew equal to the Old Value 
     * Function Guess. Only world rank 0 needs to store the complete New Value Function Matrix
     */

    
    MatrixXd ValNew=(my_rank==0) ? ValOld: MatrixXd::Zero(1,1);
    

    /* We will now do Numits - numstart Value function iterations. We begin at iteration numstart and end at iteration Numits-1.
     */
    t1=MPI_Wtime(); // Start the timing of the iterations
    for (int i=numstart; i<Numits; i++) {
        /* The ValNew resulting from the previous Value function iteration now becomes 
	 * the ValOld of the current iteration. World rank 0 process broadcasts 
	 * OldVal to all other processes and all of them wait until they have all received ValOld before proceeding.
         */
        if (my_rank==0) {
            ValOld=ValNew;
        }
        MPI_Bcast(ValOld.data(), nk*ntheta, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        
        /* Each process calls the ValIt function and stores the returned value in Result. Only process with
         world rank 0 stores the full 2nk * ntheta matrix.
         */
        
        Result=ValIt(ValOld);
        
        // Extract Value Function and Policy Function
        ValNew=(my_rank==0) ? Result.topLeftCorner(nk,ntheta) : Result;
        Policy=(my_rank==0) ? Result.bottomRightCorner(nk, ntheta) : Result;
        
        // World Rank 0 process computes and prints the maximum difference between the old and new value functions
        if (my_rank==0) {
            errmax=(ValNew-ValOld).array().abs().maxCoeff();
            printf("iteration %d max error %f \n", i+1, errmax);
            
            /* We store the results in a text file every datafreq iterations. If datafreq=1, we store the results
             for every iteration.
             */
            
            if (i % datafreq==0) {
                
                /* For each theta state we create a textfile with three columns. 
		 * The first column is the kgrid of the problem, the second column 
		 * contains the corresponding values of the value function, 
		 * and the last column contains the corresponding optimal policies
                 */
                
                for (int j=0; j<ntheta; j++) {
                    // Create text file where we will store results
                    char filename_dummy[100];
                    sprintf(filename_dummy,"%s%.2f%s%d%s", "results/theta_" , thetagrid(j), "_", i+1, ".txt");
                    output.open(filename_dummy);
                    
                    // Store data as desired in Display and print results into the text file.
                    
                    /* The display matrix stores the kgrid as its first column, the corresponding values
                     as its second column, and the optimal policies are its final column.
                     */
                    
                    Display<<kgrid, ValNew.col(j), Policy.col(j);
                    output<<" kgrid "<<" ValNew "<< " Policy " << endl;
                    output<<Display;
                    output.close();
                }
                
            }

            
        }
        
        
    }
    
    // End the timing of the iterations
    t2=MPI_Wtime();
    
    // World Rank 0 process prints the time it took to do the iterations
    if (my_rank==0) {
        
        // Print Elapsed Time
        printf( "Iterations took %f seconds\n", t2 - t1 );
        
        // Perform one last iteration so we can compute how much time it takes to write the results
        
        ValOld=ValNew;
        
    }
    
    MPI_Bcast(ValOld.data(), nk*ntheta, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    
    Result=ValIt(ValOld);
    
    
    if (my_rank==0) {
        // Start Timing the Final Part (Writing Part)
        t1=MPI_Wtime();
        
        /* Extract Value Function and Policy Function and 
	compute maximum difference between the old and new value functions*/
        
        ValNew=Result.topLeftCorner(nk,ntheta);
        Policy=Result.bottomRightCorner(nk, ntheta);
        errmax=(ValNew-ValOld).array().abs().maxCoeff();
        
        
        // Print kgrid and the corresponding Values and Policies into a text file for each theta
        
        for (int i=0; i<ntheta; i++) {
            // Create text file where we will store results
            char filename_dummy[100];
            sprintf(filename_dummy,"%s%.2f%s", "results/theta_" ,thetagrid(i),"_testing.txt");
            output.open(filename_dummy);
            
            // Store data as desired in Display and print results into the text file
            Display<<kgrid, ValNew.col(i), Policy.col(i);
            output<<" kgrid "<<" ValNew "<< " Policy " << endl;
            output<<Display;
            output.close();
        }
        
        // End timing
        t2=MPI_Wtime();
        
        printf("Final Part (writing) took %f seconds\n", t2 - t1 );
    }
    
    
    MPI_Finalize();
    
    return 0;
    
 }
 
