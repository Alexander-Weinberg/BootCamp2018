#include <iostream>
#include <math.h>
#include <map>
#include <string>
using namespace std;


pair<float,float> solve_quad(float a, float b, float c)
{
  float x1 = (- b + sqrt(b * b - (4 * a * c))) / (2 * a);
  float x2 = (- b - sqrt(b * b - (4 * a * c))) / (2 * a);

  return make_pair<float,float>(x1, x2);
}

float f(float x)
{
  float y = cos( x ) - x;
  return y;
}

float f_prime(float x)
{
  float y = -sin( x ) - 1;
  return y;
}

float newton(float (*f)(float), float (*f_prime)(float), float x0, double tol=1e-5)
{
  int maxiter = 100;
  float xlast = x0;

  float diff = 7.0;
  int its = 0;
  float xnew = 1.0;

  while (diff > tol && its < maxiter)
  {

    xnew = xlast - f(xlast) / f_prime(xlast);

    float diff = abs(xnew - xlast);
    float xlast = xnew;
    its++;
  }

return xnew;
}

int main()
{
  cout << "\n----------------------------------------------\n";

  cout << "Solving a quadratic equation\n";
  pair<double,double> result = solve_quad(1,4,-5);
  cout << "x = " << result.first << " or " << result.second << endl;

  cout << "----------------------------------------------\n";

  cout << "Find zero of function using newton's method\n";
  cout << "This is easy to do if I can hard code the derivative in.\nHow could I do this if i needed to numerically approximate derivative?\n\n";

  float root = newton(f, f_prime, 1.0);
  cout << "Root is: " << root << endl;
  cout << "----------------------------------------------\n\n";

  return 0;
}
