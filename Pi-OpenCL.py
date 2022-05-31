#
# Vadd
#
# Element wise addition of two vectors (c = a + b)
# Asks the user to select a device at runtime
#
# History: C version written by Tim Mattson, December 2009
#          C version Updated by Tom Deakin and Simon McIntosh-Smith, October 2012
#          Ported to Python by Tom Deakin, July 2013
#

# Import the Python OpenCL API
import pyopencl as cl
# Import the Python Maths Library (for vectors)
import numpy

# Import a library to print out the device information
import deviceinfo

# Import Standard Library to time the execution
from time import time

# ------------------------------------------------------------------------------

# number
N = 100000000

# ------------------------------------------------------------------------------
#
# Kernel: vectorPrime
#
# To compute is vector a prime number
#
# Input: a and b float vectors of length count
# Output c float vector of length count holding the sum a + b

kernelsource = """
__kernel void calculatePi(
    __global int* N,
    __global float* result) {
        int i = get_global_id(0);
        float sum = 0;
        float finalSum = 0.0;
        float h =  1. / N[0];
        float x =  h * (i - 0.5);
        sum = 4.0 / (1 + x*x);
        result[i] = h * sum;
}
"""

# ------------------------------------------------------------------------------

# Main procedure

# Create a compute context
# Ask the user to select a platform/device on the CLI
context = cl.create_some_context()

# Print out device info
deviceinfo.output_device_info(context.devices[0])

# Create a command queue
queue = cl.CommandQueue(context)

# Create the compute program from the source buffer
# and build it
program = cl.Program(context, kernelsource).build()

# # Create a and b vectors and fill with random float values
# h_a = numpy.random.rand(LENGTH).astype(numpy.float32)
# h_b = numpy.random.rand(LENGTH).astype(numpy.float32)
# # Create an empty c vector (a+b) to be returned from the compute device
# h_c = numpy.empty(LENGTH).astype(numpy.float32)
h_numbers = numpy.arange(N).astype(numpy.int32)
h_N = numpy.array([N]).astype(numpy.int32)
h_result = numpy.empty(N).astype(numpy.float32)

# Create the input (a, b) arrays in device memory and copy data from host
# d_a = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_a)
# d_b = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_b)
# Create the output (c) array in device memory

#d_numbers = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_numbers)
d_N = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_N)
d_result = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, h_result.nbytes)

# Start the timer
rtime = time()

# Execute the kernel over the entire range of our 1d input
# allowing OpenCL runtime to select the work group items for the device
calculatePi = program.calculatePi
calculatePi.set_scalar_arg_dtypes([None, None])
calculatePi(queue, h_numbers.shape, None, d_N, d_result)

# Wait for the commands to finish before reading back
queue.finish()
rtime = time() - rtime
print("The kernel ran in", rtime, "seconds")
# Read back the results from the compute device
cl.enqueue_copy(queue, h_result, d_result)

final_result = numpy.add.reduce(h_result)

print("Pi equals ---> " + str(final_result))