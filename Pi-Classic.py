# Import Standard Library to time the execution
from time import time

done = True
n = 1000000
sum = 0
i = 0
mypi = 0
# Start the timer
rtime = time()
for i in range(n):
    h = 1 / n
    x = h * (i - 0.5)
    sum += 4 / (1 + x*x)
    mypi = h * sum

print("The programe ran in", time() - rtime, "seconds")
print("pi is :")
print(mypi)