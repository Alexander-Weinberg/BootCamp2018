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

int main()
{
  cout << "\n----------------------------------------------\n";
  float a, b, c;
  cout << "Let's solve a quadratic equation!\n f(x) = ax^2 + bx + c = 0\n";
  cout << "This only works for equations with real solutions :(\n";
  cout << "Enter a: ";
  cin >> a;
  cout << "Enter b: ";
  cin >> b;
  cout << "Enter c: ";
  cin >> c;
  
  pair<double,double> result = solve_quad(a,b,c);
  cout << "x = " << result.first << " or " << result.second << endl;

  cout << "----------------------------------------------\n";
  return 0;
}

