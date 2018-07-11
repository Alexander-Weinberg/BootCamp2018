// my first program in C++

#include <iostream>
using namespace std; //no std:: prefix is needed!
// Hello world function

int hello() {
  std::cout << "Hello, it's Alex\n"; // print to terminal
  return 0;
}

int enter_num() {
   cout << "Enter a number: "; // ask user for a number
   int x = 0;
   cin >> x; // read number from console and store it in x
   cout << "You entered " << x << endl;
   return 0;
}

bool isEqual(int x, int y)
{
   return(x==y); // true if they are equal, else false
}

int check_equal()
{
    cout << "Enter an integer: ";
    int x;
    cin >> x;       // user enters a value

    cout << "Enter another integer: ";
    int y;
    cin >> y;

    bool equal = isEqual(x, y);     // call the above function

    if (equal)
        cout << x << " and " << y << " are equal" << endl;
    else
        cout << x << " and " << y << " are not equal" << endl;
    return 0;
}

int first_loop() {
    // for loop execution
    for( int a = 10; a < 20; a++)
    {
        cout << "value of a:" << a << endl;
    }
    return 0;
}

int main() {
    int z = hello();    // call the hello function
    int z = check_equal();
    int z = first_loop();
    return 0;
}
