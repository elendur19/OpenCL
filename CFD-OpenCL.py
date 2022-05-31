import math
import sys
from locale import atoi
# Import Standard Library to time the execution
from time import time
# Import the Python OpenCL API
import pyopencl as cl
# Import the Python Maths Library (for vectors)
import numpy

# Import a library to print out the device information
import deviceinfo

# ------------------------------------------------------------------------------
#
# Kernel: iterative Jacobi loop
#

kernelsource = """
__kernel void calculateBnorm(
    const unsigned int m,
    const unsigned int n,
    __global double* psi,
    __global double* bnorm) {
    int id = get_global_id(0);
    int nbrOfProcs = get_global_size(0);
    int i, j;
    for (i = id; i < m + 2; i += nbrOfProcs) {
            for (j = 0; j < n + 2; j++) {
                bnorm[i] = psi[i*(m+2)+j]*psi[i*(m+2)+j];
        }
    } 
}



__kernel void jacobistep(
    const unsigned int m,
    const unsigned int n,
    __global double* psi,
    __global double* psitmp) {
    int id = get_global_id(0);
    int nbrOfProcs = get_global_size(0);
    int i, j;
    for(i = id + 1; i <= m; i += nbrOfProcs) {
        for(j = 1; j <= n; j++) {
            psitmp[i*(m+2)+j] = 0.25 * (psi[(i-1)*(m+2)+j] + psi[(i+1)*(m+2)+j] + psi[i*(m+2)+j-1] + psi[i*(m+2)+j+1]);
        }
    }
}

__kernel void deltasq(
    const unsigned int m,
    const unsigned int n,
    __global double* psi,
    __global double* psitmp,
    __global double* error) {
    int id = get_global_id(0);
    int nbrOfProcs = get_global_size(0);
    int i, j;
    
    float tmp;
    
    for(i = id + 1; i <= m; i += nbrOfProcs) {
        for(j = 1; j <= n; j++) {
            tmp = psitmp[i * (m+2) + j] - psi[i * (m+2) + j];
            error[i * (m+2) + j] = tmp * tmp;
        }
    }
}

__kernel void copy_psi_psitmp(
    const unsigned int m,
    const unsigned int n,
    __global double* psi,
    __global double* psitmp) {
    int id = get_global_id(0);
    int nbrOfProcs = get_global_size(0);
    int i, j;
    //copy back
    for(i = id + 1; i <= m; i += nbrOfProcs) {
        for(j = 1; j <= n; j++) {
            psi[i * (m+2) + j] = psitmp[i * (m+2) + j];
        }
    }
}

"""

# ------------------------------------------------------------------------------

def boundarypsi(psi, m, n, b, h, w):
    # BCs on bottom edge
    for i in range(b+1, b+w):
        psi[i*(m+2)+0] = (i-b)

    for i in range(b + w, m + 1):
        psi[i*(m+2)+0] = float(w)

    # BCS on RHS
    for j in range(1, h + 1):
        psi[(m+1)*(m+2)+j] = float(w)

    for j in range(h+1, h+w):
        psi[(m+1)*(m+2)+j] = float(w-j+h)

# output
printfreq = 1000
# tolerance for convergence. <= 0 means do not check
tolerance = 0.0

# command line arguments

# simulation sizes
bbase = 10
hbase = 15
wbase = 5
mbase = 32
nbase = 32

irrotational = 1
checkerr = 0

# do we stop because of tolerance?
if tolerance > 0:
    checkerr = 1

# check command line parameters and parse them

if len(sys.argv) < 3 or len(sys.argv) > 4:
    print("Usage: cfd <scale> <numiter>\n")


scalefactor = atoi(sys.argv[1])
numiter = atoi(sys.argv[2])

if checkerr == 0:
    print("Scale Factor = {0}, iterations = {1}\n".format(scalefactor, numiter))
else:
    print("Scale Factor = %i, iterations = %i, tolerance= %g\n", scalefactor, numiter, tolerance)

print("Irrotational flow\n")

# Calculate b, h & w and m & n
b = bbase * scalefactor
h = hbase * scalefactor
w = wbase * scalefactor
m = mbase * scalefactor
n = nbase * scalefactor

print("Running CFD on {0} x {1} grid in serial\n".format(m, n))

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

# allocate arrays
h_psi = numpy.zeros((m + 2) * (n + 2)).astype(numpy.double)
h_psitmp = numpy.empty((m + 2) * (n + 2)).astype(numpy.double)
h_error = numpy.zeros(h_psitmp.size).astype(numpy.double)
h_bnorm = numpy.empty(m + 2).astype(numpy.double)

# set the psi boundary conditions
boundarypsi(h_psi, m, n, b, h, w)

bnorm = 0.0

# compute normalisation factor for error with kernel function

d_bnorm = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, h_bnorm.nbytes)
d_psi = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_psi)

# calculateBnorm = program.calculateBnorm
# calculateBnorm.set_scalar_arg_dtypes([numpy.uint32, numpy.uint32, None, None])
# calculateBnorm(queue, h_bnorm.shape, None, m, n, d_psi, d_bnorm)
#
# # Wait for the commands to finish before reading back
# queue.finish()
# cl.enqueue_copy(queue, h_bnorm, d_bnorm)
#
# bnorm = numpy.add.reduce(h_bnorm)

for i in range(0, m + 2):
    for j in range(0, n + 2):
        bnorm += h_psi[i * (m+2) + j] * h_psi[i * (m+2) + j]

bnorm = math.sqrt(bnorm)

# begin iterative Jacobi loop
print("\nStarting main loop...\n\n")
# Start the timer
rtime = time()

d_psitmp = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_psitmp)
d_error = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, h_error.nbytes)

jacobistep = program.jacobistep
jacobistep.set_scalar_arg_dtypes([numpy.uint32, numpy.uint32, None, None])

deltasq = program.deltasq
deltasq.set_scalar_arg_dtypes([numpy.uint32, numpy.uint32, None, None, None])

copy_psi_psitmp = program.copy_psi_psitmp
copy_psi_psitmp.set_scalar_arg_dtypes([numpy.uint32, numpy.uint32, None, None])

# for psi in h_psi:
#     if psi != 0.0:
#         print("Psi value --> {0}".format(psi))

error = 0

for iter in range(1, numiter + 1):
    # calculate psi for next iteration
    jacobistep(queue, (m,), None, m, n, d_psi, d_psitmp)
    # Wait for the commands to finish before reading back
    queue.finish()
    #cl.enqueue_copy(queue, h_psitmp, d_psitmp)

    # calculate current error if required
    if checkerr  or iter == numiter:
        deltasq(queue, (m,), None, m, n, d_psi, d_psitmp, d_error)
        # Wait for the commands to finish before reading back
        queue.finish()
        cl.enqueue_copy(queue, h_error, d_error)
        error = numpy.add.reduce(h_error)
        error = math.sqrt(error)
        error = error / bnorm

    # quit early if we have reached required tolerance
    if checkerr:
        if error < tolerance:
            print("Converged on iteration %d\n", iter)
            break

    copy_psi_psitmp(queue, (m,), None, m, n, d_psi, d_psitmp)

    # print loop information
    if iter % printfreq == 0:
        if not checkerr:
            print("Completed iteration {0}\n".format(iter))
        else:
            print("Completed iteration {0}, error = {1}\n".format(iter, error))


if iter > numiter:
    iter = numiter

# total time
rtime = time() - rtime

titer = rtime / float(iter)

# print out some stats
print("\n... finished\n")
print("After {0} iterations, the error is {1}\n".format(iter, error))
print("Time for {0} iterations was {1} seconds\n".format(iter, rtime))

