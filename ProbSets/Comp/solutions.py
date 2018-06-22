# solutions.py
"""Volume IB: Testing.
<Name>
<Date>
"""
import math
from itertools import combinations
import numpy as np

# Problem 1 Write unit tests for addition().
# Be sure to install pytest-cov in order to see your code coverage change.

def smallest_factor(n):
    """Finds the smallest prime factor of a number.
    Assume n is a positive integer.
    I added in +1 to the range
    """
    if n == 1:
        return 1
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return i
    return n


# Problem 2 Write unit tests for operator().
def month_length(month, leap_year=False):
    """Return the number of days in the given month."""
    if month in {"September", "April", "June", "November"}:
        return 30
    elif month in {"January", "March", "May", "July",
                   "August", "October", "December"}:
        return 31
    if month == "February":
        if not leap_year:
            return 28
        else:
            return 29
    else:
        return None

# Problem 3 Write unit test for this class.
def operate(a, b, oper):
    """Apply an arithmetic operation to a and b."""
    if type(oper) is not str:
        raise TypeError("oper must be a string")
    elif oper == '+':
        return a + b
    elif oper == '-':
        return a - b
    elif oper == '*':
        return a * b
    elif oper == '/':
        if b == 0:
            raise ZeroDivisionError("division by zero is undefined")
        return a / b
    else:
        raise ValueError("oper must be one of '+', '/', '-', or '*' ")

class Fraction(object):
    """Reduced fraction class with integer numerator and denominator."""
    def __init__(self, numerator, denominator):
        if denominator == 0:
            raise ZeroDivisionError("denominator cannot be zero")
        elif type(numerator) is not int or type(denominator) is not int:
            raise TypeError("numerator and denominator must be integers")

        def gcd(a,b):
            while b != 0:
                a, b = b, a % b
            return a
        common_factor = gcd(numerator, denominator)
        self.numer = numerator // common_factor
        self.denom = denominator // common_factor

    def __str__(self):
        if self.denom != 1:
            return "{} / {}".format(self.numer, self.denom).replace(" ","")
        else:
            return str(self.numer)

    def __float__(self):
        return self.numer / self.denom

    def __eq__(self, other):
        if type(other) is Fraction:
            return self.numer==other.numer and self.denom==other.denom
        else:
            return float(self) == other

    def __add__(self, other):
        return Fraction(self.numer*other.denom + self.denom*other.numer,
                                                    self.denom*other.denom)

    def __sub__(self, other):
        return Fraction(self.numer*other.denom - self.denom*other.numer,
                                                        self.denom*other.denom)

    def __mul__(self, other):
        return Fraction(self.numer*other.numer, self.denom*other.denom)

    def __truediv__(self, other):
        if self.denom*other.numer == 0:
            raise ZeroDivisionError("cannot divide by zero")
        return Fraction(self.numer*other.denom, self.denom*other.numer)


# Question 5
def count_sets(cards):
    """Return the number of sets in the provided Set hand.
    Parameters:
    cards (list(str)) a list of twelve cards as 4-bit integers in
    base 3 as strings, such as ["1022", "1122", ..., "1020"].
    Returns:
    (int) The number of sets in the hand.
    Raises:
    ValueError: if the list does not contain a valid Set hand, meaning
    - there are not exactly 12 cards,
    - the cards are not all unique,
    - one or more cards does not have exactly 4 digits, or
    - one or more cards has a character other than 0, 1, or 2.
    """

    if len(cards) != 12:
        raise ValueError("there are not 12 cards")

    if len(set(cards)) != 12:
        raise ValueError("cards are not unique")

    for card in cards:
        if len(card) != 4:
            raise ValueError("card does not have exactly 4 digits")

    for card in cards:
        for num in card:
            if num not in ['0','1','2']:
                raise ValueError('contains character other than 0,1,2')


    combos = list(combinations(cards,3))
    num_sets = np.zeros(len(combos))

    for i,val in enumerate(combos):
        if is_set(*val):
            num_sets[i] = 1

    final_val = np.sum(num_sets)
    return int(final_val)


def is_set(a, b, c):
    """Determine if the cards a, b, and c constitute a set.
    Parameters:
        a, b, c (str): string representations of 4-bit integers in base 3.
        For example, "1022", "1122", and "1020" (which is not a set).
    Returns:
        True if a, b, and c form a set, meaning the ith digit of a, b, and c are
        either the same or all different for i=1,2,3,4.

        False if a, b, and c do not form a set.
    """
    color = [a[0], b[0], c[0]]
    shape = [a[1], b[1], c[1]]
    pattern = [a[2], b[2], c[2]]
    quantity = [a[3], b[3], c[3]]

    # Thank you to Rebekah for idea

    w,x,y,z = False,False,False,False

    if len(set(color)) == 3 or len(set(color)) == 1:
        w = True
    if len(set(shape)) == 3 or len(set(shape)) == 1:
        x = True
    if len(set(pattern)) == 3 or len(set(pattern)) == 1:
        y = True
    if len(set(quantity)) == 3 or len(set(quantity)) == 1:
        z = True

    if w & x & y & z:
        return True
    else:
        return False


hand = ["1022", "1122", "0100", "2021",
"0010", "2201", "2111", "0020",
"1102", "0210", "2110", "1020"]

set1 = ['1022', '1122', '1222']
print(is_set(*set1))
print(count_sets(hand))
