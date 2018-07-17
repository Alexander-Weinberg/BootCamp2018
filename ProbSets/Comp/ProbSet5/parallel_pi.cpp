#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
 
int main()
{
    const int niter = 10000000;
    double x,y;
    int count=0;
    double z;
    double pi;
    int threadID;
 
    //main loop
    #pragma omp parallel private(x, y, z, threadID)
    {
      threadID = omp_get_thread_num();
      srand(threadID);
    #pragma omp for reduction(+:count)
    for (int i=0; i<niter; ++i) {   
        //get random points
        x = (double)random()/RAND_MAX;
        y = (double)random()/RAND_MAX;
        z = sqrt((x*x)+(y*y));
        //check to see if point is in unit circle
        if (z<=1)
        {
            ++count;
        }
    }
    /* end omp parallel */
    }
    pi = ((double)count/(double)niter)*4.0;          //p = 4(m/n)
    printf("Pi: %f\n", pi);
    return 0;
}
