/*
A function to estimate pi. Uses the monte carlo method to draw random samples
(x,y) from [0,1]x[0,1] and checks if (x,y) is in unit circle.

estimate_pi = #inCircle / #totaldraws
*/

#include <random>
#include <stdio.h>

#include <iostream>
using namespace std;

bool inCircle(double x, double y)
{
	double z;
	z = x * x + y * y;
	return (z <= 1);
}


int main()
{
	unsigned seed_unif1 = 3; // unsigned means non-negative, declare seed

	std::default_random_engine generator_unif(seed_unif1);
	std::uniform_real_distribution<double> distribution_unif(0.0,1.0);

	int total_draws = 19000;
	int N_inCircle = 0;
	double randX;
	double randY;


// Generate (x,y) points
	for(int numbers = 1; numbers <= total_draws; numbers++)
	{
	  randX = distribution_unif(generator_unif); // draw from unif
		randY = distribution_unif(generator_unif); // draw from unif

		bool inside = inCircle(randX, randY); // check in unit circle

		if (inside)
				N_inCircle += 1;
	}

// Compute estimate of pi
	double estimate;
	estimate = 4 * (double) N_inCircle / (double)total_draws;
	cout << "My estimate of pi is:  " << estimate << "\t with samples = " << total_draws<<endl;
	cout << "Answer should be near: 3.141\n";
	cout << "Requires 19_000 samples to get first two decimals accurate\n";

  return 0;
}
