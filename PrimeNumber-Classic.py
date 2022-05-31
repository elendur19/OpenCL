# Import the Python OpenCL API
import pyopencl as cl
# Import the Python Maths Library (for vectors)
import numpy

# Import a library to print out the device information
from numpy import double
import math
import deviceinfo

# Import Standard Library to time the execution
from time import time
from math import sqrt
# Number of examples
N = 1000000
# Number of prime numbers
counter = 0
# Start the timer
rtime = time()
for number in range(N):
    prime_flag = 0
    if number > 1:
        for j in range(2, math.isqrt(number)):
            # If number is divisible by any number between 2 and n / 2 it is not prime
            if (number % j) == 0:
                prime_flag = 1
                break
        if prime_flag == 0:
            counter += 1

print("The programe ran in", time() - rtime, "seconds")
print(counter)