# test_solutions.py
"""Volume 1B: Testing.
<Name>
<Class>
<Date>
"""

import solutions as soln
from solutions import is_set
import pytest

# Problem 1: Test the addition and fibonacci functions from solutions.py
def test_addition():
    pass

def test_smallest_factor():
    assert soln.smallest_factor(4) == 2, 'Range is too small'
    assert soln.smallest_factor(6) == 2, 'Range is too small'
    assert soln.smallest_factor(8) == 2, 'Range is too small'

    assert soln.smallest_factor(1) == 1, 'One fails'
    assert soln.smallest_factor(5) == 5, 'Primes fail'

def better_smallest_factor(n):
    """Finds the smallest prime factor of a number.
    Assume n is a positive integer.
    """
    if n == 1:
        return 1
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return i
    return n

# Problem 2: Test the operator function from solutions.py
def test_month_length():
    assert soln.month_length('September') == 30, '30 day months fail'
    assert soln.month_length('January') == 31, '31 day months fail'
    assert soln.month_length('February',leap_year=False) == 28, 'Leap february fail'
    assert soln.month_length('February',leap_year=True) == 29, 'Leap february fail'
    assert soln.month_length('Vegan chocolate') == None, 'Non months fail'


# Problem 3: Finish testing the complex number class
def test_operate():
    assert soln.operate(10,2,'+') == 12, 'addition fails'
    assert soln.operate(10,2,'-') == 8, 'subtraction fails'
    assert soln.operate(10,2,'/') == 5, 'division fails'
    assert soln.operate(10,2,'*') == 20, 'multiply fails'

    with pytest.raises(TypeError) as excinfo:
        soln.operate(1,1,5)
    with pytest.raises(ZeroDivisionError) as excinfo:
        soln.operate(1,0,'/')
    with pytest.raises(ValueError) as excinfo:
        soln.operate(1,1,'Math frat')


@pytest.fixture
def set_up_fractions():
    frac_1_3 = soln.Fraction(1, 3)
    frac_1_2 = soln.Fraction(1, 2)
    frac_n2_3 = soln.Fraction(-2, 3)
    return frac_1_3, frac_1_2, frac_n2_3

def test_fraction_init(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert frac_1_3.numer == 1
    assert frac_1_2.denom == 2
    assert frac_n2_3.numer == -2
    frac = soln.Fraction(30, 42) # 30/42 reduces to 5/7.
    assert frac.numer == 5
    assert frac.denom == 7

def test_fraction_str(set_up_fractions):
    ''' I added in .strip() '''
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert str(frac_1_3) == "1/3"
    assert str(frac_1_2) == "1/2"
    assert str(frac_n2_3) == "-2/3"

def test_fraction_float(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert float(frac_1_3) == 1 / 3.
    assert float(frac_1_2) == .5
    assert float(frac_n2_3) == -2 / 3.

def test_fraction_eq(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert frac_1_2 == soln.Fraction(1, 2)
    assert frac_1_3 == soln.Fraction(2, 6)
    assert frac_n2_3 == soln.Fraction(8, -12)

def test_init():
    with pytest.raises(ZeroDivisionError) as excinfo:
        soln.Fraction.__init__('hey', 5, 0)

    with pytest.raises(TypeError) as excinfo:
        soln.Fraction.__init__('hey', 5.4, 3.6)

def test_str():
    test_frac = soln.Fraction(5,1)
    assert test_frac.__str__() == str(test_frac.numer) , 'denom = 1 fails'

def test_eq():
    test_frac = soln.Fraction(1,3)
    test2_frac = soln.Fraction(1,2)

    assert test_frac.__eq__(test_frac) == True
    assert test_frac.__eq__(5) == False

def test_add():
    frac_1_3, frac_1_2 = soln.Fraction(1,3), soln.Fraction(1,2)
    assert soln.Fraction.__add__(frac_1_3, frac_1_2) == soln.Fraction(5,6)

def test_sub():
    test_frac = soln.Fraction(2,3)
    test2_frac = soln.Fraction(1,3)
    assert test_frac.__sub__(test2_frac) == soln.Fraction(1,3)

def test_mul():
    test_frac = soln.Fraction(2,3)
    test2_frac = soln.Fraction(1,3)
    assert test_frac.__mul__(test2_frac) == soln.Fraction(2,9)

def test_truediv():
    test_frac = soln.Fraction(1,3)
    test2_frac = soln.Fraction(2,3)
    test3_frac = soln.Fraction(0, 3)
    assert test_frac.__truediv__(test2_frac) == soln.Fraction(1,2)

    with pytest.raises(ZeroDivisionError) as excinfo:
        test_frac.__truediv__(test3_frac)

# Question 5, set game

def test_isSet():
    ''' There are 5 kinds of sets.
    ==============================================================
    (a) Same in quantity and shape; different in pattern and color
    (b) Same in color and pattern; different in shape and quantity
    (c) Same in pattern; different in shape, quantity and color
    (d) Same in shape; different in quantity, pattern and color
    (e) Different in all aspects
    ==============================================================
    0,1,2 = red, green, purple
    0,1,2 = one, two , three
    0,1,2 = empty, striped, filled
    0,1,2 = squigle, oval, diamond

    '''
    assert is_set("0101","1111","2121") == True
    assert is_set("1010","1111","1212") == True
    assert is_set("0010","1111","2212") == True
    assert is_set("0001","1111","2221") == True
    assert is_set("0000","1111","2222") == True

    return

def test_count_sets():
    hand = ["1022", "1122", "0100", "2021",
    "0010", "2201", "2111", "0020",
    "1102", "0210", "2110", "1020"]

    notunique = ["0000", "0000", "0000", "0000", "0000", "0000", "0000","0000",
            "0000","0000","0000","0000"]

    not12 = ["0000"]

    not4 = ["0", "0000", "2121", "1010", "1111", "1212", "0010","1110",
            "2212","0001","2222","2221"]

    has3 = ["0103", "0000", "2121", "1010", "1111", "1212", "0010","1110",
            "2212","0001","2222","2221"]

    assert soln.count_sets(hand) == 6

    with pytest.raises(ValueError) as excinfo:
        soln.count_sets(notunique)

    with pytest.raises(ValueError) as excinfo:
        soln.count_sets(not12)

    with pytest.raises(ValueError) as excinfo:
        soln.count_sets(not4)

    with pytest.raises(ValueError) as excinfo:
        soln.count_sets(has3)

    return
